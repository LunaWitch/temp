import numpy as np
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

    generations = system_config["NUM_GENERATION"]

    checkpoint = None
    for generation in range(generations):
        dataset = collect_episode(checkpoint, system_config, user_config)
        dataset = preprocess_data(dataset, checkpoint, system_config, user_config)
        trainer = get_trainer(dataset, checkpoint, system_config, user_config)
        result = trainer.fit()
        checkpoint = save_latest_model(result, system_config, user_config)
        if validate_model(checkpoint, system_config, user_config) :
            break
        
def collect_episode(checkpoint, system_config, user_config):
    print("collect_episode")
    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    num_actor = ray_config["NUM_ACTOR"]
    num_episode = train_config["NUM_EPISODE"]

    if checkpoint:
        with checkpoint.as_directory() as CHECK_POINT_DIR:
            model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
    else :
        model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]
    actors = [Actor.remote(model_path, user_config, system_config) for _ in range(num_actor)]
    ray.get([actor.ready.remote() for actor in actors])

    num_trajectory = train_config["NUM_TRAJECTORY"]
    pending = []
    for i in range(num_episode):
        actor = actors[i % num_actor]
        pending.append(actor.get_episode.remote(num_trajectory))
    results = ray.get(pending)
    all_training_data = [item for result in results for item in result]
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

def preprocess_data(dataset, checkpoint, system_config, user_config):
    print("preprocess_data")
    train_config = system_config["TRAIN"]
    ray_config = system_config["RAY"]
    if checkpoint:
        with checkpoint.as_directory() as CHECK_POINT_DIR:
            model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
    else :
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

        processed = worker.preprocess_data(state, next_state, action, reward, log_prob, done)

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
    system_config = config["SYSTEM_CONFIG"]
    user_config = config["USER_CONFIG"]
    checkpoint = config.get("CHECKPOINT")

    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    wandb_config = system_config["WANDB"]

    if checkpoint:
        with checkpoint.as_directory() as CHECK_POINT_DIR:
            model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
    else:
        model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]
    worker = Worker(model_path, user_config, system_config)
    worker.prepare_model()
    worker_id = session.get_world_rank()
    use_wandb = wandb_config["ENABLE"] and worker_id == 0
    if use_wandb:
        wandb.require("core")
        wandb.init(
            project=wandb_config["PROJECT_NAME"],
            name=f"{session.get_experiment_name()}",
            reinit=True,
        )
        wandb.watch(worker.get_model_list(), log="all", log_freq=100)

    num_epoch = train_config["NUM_EPOCH"]
    batch_size = train_config["BATCH_SIZE"]
    dataset = session.get_dataset_shard("train")

    for epoch in range(num_epoch):
        cumulative = defaultdict(float)
        count = defaultdict(int)
        for batch in dataset.iter_torch_batches(batch_size=batch_size):
            float_keys = ["state", "next_state", "action", "reward", "log_prob", "done"]
            tensors = {k: batch[k].float() for k in float_keys}
            preprocess = {
                k: v.float() for k, v in batch.items() if k not in float_keys
            }

            metrics = worker.train_model(
                tensors["state"],
                tensors["next_state"],
                tensors["action"],
                tensors["reward"],
                tensors["log_prob"],
                tensors["done"],
                preprocess,
            )

            for k, v in metrics.items():
                cumulative[k] += v.item()
                count[k] += 1

        # 평균 계산 및 로그 기록
        avg_metric = {k: cumulative[k] / count[k] for k in cumulative}
        avg_metric["epoch"] = (epoch + 1) / num_epoch

        if use_wandb:
            wandb.log(avg_metric)

        # Checkpoint 저장 (마지막 epoch만)
        if epoch == num_epoch - 1:
            with TemporaryDirectory() as temp:
                temp_path = Path(temp) / ray_config["CHECKPOINT_MODEL_NAME"]
                worker.save_model(temp_path)
                checkpoint = Checkpoint.from_directory(temp)
                train.report(metrics=avg_metric, checkpoint=checkpoint)
        else:
            train.report(metrics=avg_metric)

    if use_wandb:
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

def validate_model(checkpoint, system_config, user_config):
    print("validate_model")
    ray_config = system_config["RAY"]
    train_config = system_config["TRAIN"]
    
    validate_config = system_config["VALIDATE"]
    num_actor = ray_config["NUM_ACTOR"]
    num_episode = validate_config["NUM_EPISODE"]
    if checkpoint:
        with checkpoint.as_directory() as CHECK_POINT_DIR:
            model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
    else :
        model_path = RESULT_DIR / train_config["LATEST_MODEL_NAME"]
    actors = [Actor.remote(model_path, user_config, system_config) for _ in range(num_actor)]
    ray.get([actor.ready.remote() for actor in actors])

    num_trajectory = validate_config["NUM_TRAJECTORY"]
    pending = []
    for i in range(num_episode):
        actor = actors[i % num_actor]
        pending.append(actor.get_score.remote(num_trajectory))
    
    results = ray.get(pending)
    score = np.mean(results)
    print(f"Result {score}")
    return score >= validate_config["SCORE"]