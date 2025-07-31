import numpy as np
from torch.distributed import is_initialized, destroy_process_group
import yaml
import wandb

import shutil
import ray
from ray import train
from ray.data import from_items
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.train import RunConfig
from ray.train import CheckpointConfig
from ray.train import FailureConfig
from ray.train import Checkpoint
from ray.air import session


from tempfile import TemporaryDirectory
from tqdm import tqdm
from collections import defaultdict

from Trainer.actor import Actor
from Trainer.worker import Worker
from pathlib import Path
from util.path_util import get_config_dir, get_user_define_dir, get_result_dir
from user_define.model import ModelWrapper

CONFIG_DIR = get_config_dir()
USER_DEFINE_DIR = get_user_define_dir()
RESULT_DIR = get_result_dir()
TRAIN_YAML_FILE = "train.yaml"


def run(user_config_name):
    with open(CONFIG_DIR / TRAIN_YAML_FILE, "r") as f:
        config = yaml.safe_load(f)
        system_config = config

    with open(USER_DEFINE_DIR / user_config_name, "r") as f:
        user_config = yaml.safe_load(f)

    generations = system_config["TRAIN"]["NUM_GENERATION"]

    prev_checkpoint = None
    for generation in range(generations):
        dataset = collect_episode(system_config, user_config)
        dataset = preprocess_data(dataset, system_config, user_config)
        #dataset = dataset.random_shuffle()
        #dataset = dataset.repartition(system_config["RAY"]["NUM_TRAINER"])
        trainer = get_trainer(dataset, prev_checkpoint, system_config, user_config)
        result = trainer.fit()
        prev_checkpoint = save_latest_model(result, system_config, user_config)
        # TODO Validation -> Log / Early Stop


def collect_episode(system_config, user_config):
    print("collect_episode")
    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    num_actor = ray_config["NUM_ACTOR"]
    num_episode = train_config["NUM_EPISODE"]
    episode_per_actor = num_episode // num_actor

    latest_model_name = train_config["LATEST_MODEL_NAME"]
    model_path = RESULT_DIR / latest_model_name
    actors = [Actor.remote(model_path, user_config, system_config) for _ in range(num_actor)]
    ray.get([actor.ready.remote() for actor in actors])

    num_trajectory = train_config["NUM_TRAJECTORY"]
    pending = [
        actor.get_episode.remote(num_trajectory)
        for actor in actors
        for _ in range(episode_per_actor)
    ]

    all_training_data = []
    for _ in tqdm(range(len(pending))):
        done, pending = ray.wait(pending, num_returns=num_actor)
        results = ray.get(done)
        for data in results:
            all_training_data.extend(data)

    dataset = from_items([
        {
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "log_prob": log_prob,
            "done": done,
        }
        for state, next_state, action, reward, log_prob, done in all_training_data
    ])
    return dataset


def preprocess_data(dataset, system_config, user_config):
    print("preprocess_data")
    train_config = system_config["TRAIN"]
    model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]
    worker = Worker(model_path, user_config, system_config)

    new_items = []
    for batch in dataset.iter_batches(batch_size=1024):
        state = np.stack(batch["state"])
        next_state = np.stack(batch["next_state"])
        action = np.stack(batch["action"])
        reward = np.stack(batch["reward"])
        log_prob = np.stack(batch["log_prob"])
        done = np.stack(batch["done"])

        processed = worker.preprocess_data(
            state, next_state, action, reward, log_prob, done
        )

        for i in range(len(state)):
            item = {key: batch[key][i] for key in batch}
            for key in processed:
                item[key] = processed[key][i]
            new_items.append(item)
    return from_items(new_items)


def get_trainer(dataset, prev_checkpoint, system_config, user_config):
    print("get_trainer")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "CHECKPOINT": prev_checkpoint,
            "SYSTEM_CONFIG": system_config,
            "USER_CONFIG": user_config,
        },
        datasets={"train": dataset},
        scaling_config=ScalingConfig(
            use_gpu=True,
            num_workers=system_config["RAY"]["NUM_TRAINER"],
            resources_per_worker={"GPU": 1},
        ),
        run_config=RunConfig(
            name=f"train_model",
            storage_path=RESULT_DIR,
            # failure_config = FailureConfig(max_failures = -1),
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    return trainer


def train_loop_per_worker(config):
    print("train_loop")
    system_config = config.get("SYSTEM_CONFIG")
    user_config = config.get("USER_CONFIG")
    checkpoint = config.get("CHECKPOINT")

    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    wandb_config = system_config["WANDB"]

    model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]

    worker = Worker(model_path, user_config, system_config)
    worker.prepare_model()

    worker_id = session.get_world_rank()
    if wandb_config["ENABLE"] and worker_id == 0:
        wandb.require("core")
        wandb.init(
            project=wandb_config["PROJECT_NAME"],
            name=f"{session.get_experiment_name()}",
            reinit=True,
        )
        wandb.watch((worker.model.actor, worker.model.critic), log="all", log_freq=100)

    num_epoch = train_config["NUM_EPOCH"]

    dataset = session.get_dataset_shard("train")
    for epoch in range(num_epoch):
        batch_iterator = dataset.iter_torch_batches(batch_size=train_config["BATCH_SIZE"])
        cumulative = defaultdict(float)
        cumulative_count = defaultdict(int)
        for batch in batch_iterator:
            batch_state = batch["state"].float()
            batch_action = batch["action"].float()
            batch_reward = batch["reward"].float()
            batch_next_state = batch["next_state"].float()
            batch_log_prob = batch["log_prob"].float()
            batch_done = batch["done"].float()
            metric = worker.train_model(batch_state, batch_next_state, batch_action, batch_reward, batch_log_prob, batch_done, batch)
            for k, v in metric.items():
                cumulative[k] += v.item()
                cumulative_count[k] += 1

        avg_metric = {k: cumulative[k] / cumulative_count[k] for k in cumulative}
        avg_metric["epoch"] = (epoch + 1) / num_epoch
        if wandb_config["ENABLE"] and worker_id == 0:
            wandb.log(avg_metric)
        
        with TemporaryDirectory() as temp:
            temp_path = Path(temp) / ray_config["CHECKPOINT_MODEL_NAME"]
            worker.save_model(temp_path)
            checkpoint = Checkpoint.from_directory(temp)
            train.report(metrics=avg_metric, checkpoint=checkpoint)

    if wandb_config["ENABLE"] and worker_id == 0:
        wandb.finish()

def save_latest_model(result, system_config, user_config):
    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    if result.best_checkpoints:
        best_checkpoint = result.best_checkpoints[0][0]
        with best_checkpoint.as_directory() as CHECK_POINT_DIR:
            source_model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
            latest_model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]
            shutil.copy(source_model_path, latest_model_path)
            print(f"Training finished. New model saved to {latest_model_path}")
        return best_checkpoint
    return None
