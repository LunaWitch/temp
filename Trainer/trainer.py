
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
from collections import defaultdict

from Trainer.actor import Actor
from Trainer.worker import Worker
from pathlib import Path
from util.path_util import get_result_dir

RESULT_DIR = get_result_dir()

class Trainer:
    model_name = {}
    model_index = 0
    checkpoint = {}
    def __init__(self, system_config):
        self.system_config = system_config

    def run(self, user_config): 
        name = user_config["NAME"]
        if name not in Trainer.model_name:
            Trainer.checkpoint[name] = None
            Trainer.model_name[name] = self.model_index
            Trainer.model_index += 1
        dataset = self.collect_episode(self.checkpoint[name], self.system_config, user_config)
        trainer = self.get_trainer(dataset, self.checkpoint[name], self.system_config, user_config)
        result = trainer.fit()
        self.checkpoint[name] = self.save_latest_model(result, self.system_config, user_config)
        if self.validate_model(self.checkpoint[name], self.system_config, user_config) :
            return True
        return False

    def collect_episode(self, checkpoint, system_config, user_config):
        print(f"collect_episode")
        ray_config = system_config["RAY"]
        train_config = system_config["TRAIN"]
        num_actor = ray_config["NUM_ACTOR"]
        num_episode = train_config["NUM_EPISODE"]

        if checkpoint:
            with checkpoint.as_directory() as CHECK_POINT_DIR:
                model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
        else :
            model_path = RESULT_DIR / user_config["NAME"] / train_config["LATEST_MODEL_NAME"]
        actors = [Actor.remote(user_config, system_config) for _ in range(num_actor)]
        ray.get([actor.ready.remote(model_path) for actor in actors])

        num_trajectory = train_config["NUM_TRAJECTORY"]
        pending = []
        for i in range(num_episode):
            actor = actors[i % num_actor]
            pending.append(actor.get_episode.remote(num_trajectory))
        results = ray.get(pending)
        all_training_data = [item for result in results for item in result]
        dataset = from_items(all_training_data)
        return dataset

    def get_trainer(self, dataset, prev_checkpoint, system_config, user_config):
        print(f"get_trainer")
        trainer = TorchTrainer(
            train_loop_per_worker=self.train_loop_per_worker,
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
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute="loss",
                    checkpoint_score_order="min",
                ),
            ),
        )
        return trainer

    def train_loop_per_worker(self, config):
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
            model_path = RESULT_DIR / user_config["NAME"] / train_config["LATEST_MODEL_NAME"]
        worker = Worker(user_config, system_config)
        worker.ready(model_path)
        worker_id = session.get_world_rank()
        use_wandb = wandb_config["ENABLE"] and worker_id == 0
        if use_wandb:
            wandb.init(project = wandb_config["PROJECT_NAME"], name = session.get_experiment_name(), reinit=True)
            wandb.watch(worker.get_model_list(), log="all", log_freq=100)

        num_epoch = train_config["NUM_EPOCH"]
        batch_size = train_config["BATCH_SIZE"]
        dataset = session.get_dataset_shard("train")

        for epoch in range(num_epoch):
            cumulative = defaultdict(float)
            count = defaultdict(int)
            for batch in dataset.iter_torch_batches(batch_size=batch_size):
                metrics = worker.train_model(batch)

                for k, v in metrics.items():
                    cumulative[k] += v.item()
                    count[k] += 1
            avg_metric = {k: cumulative[k] / count[k] for k in cumulative}
            avg_metric["epoch"] = (epoch + 1) / num_epoch

            if use_wandb:
                wandb.log(avg_metric)

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

    def save_latest_model(self, result, system_config, user_config):
        ray_config = system_config["RAY"]
        train_config = system_config["TRAIN"]
        if result.best_checkpoints:
            best_checkpoint = result.best_checkpoints[0][0]
            with best_checkpoint.as_directory() as CHECK_POINT_DIR:
                source_model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
                latest_model_path = RESULT_DIR / user_config["NAME"] / train_config["LATEST_MODEL_NAME"]
                latest_model_path.parent.mkdir(parents=True, exist_ok=True) 
                shutil.copy(source_model_path, latest_model_path)
                print(f"Training finished. New model saved to {latest_model_path}")
            return best_checkpoint
        return None

    def validate_model(self, checkpoint, system_config, user_config):
        print(f"validate_model")
        ray_config = system_config["RAY"]
        train_config = system_config["TRAIN"]
        
        validate_config = system_config["VALIDATE"]
        num_actor = ray_config["NUM_ACTOR"]
        num_episode = validate_config["NUM_EPISODE"]
        if checkpoint:
            with checkpoint.as_directory() as CHECK_POINT_DIR:
                model_path = Path(CHECK_POINT_DIR) / ray_config["CHECKPOINT_MODEL_NAME"]
        else :
            model_path = RESULT_DIR / user_config["NAME"] / train_config["LATEST_MODEL_NAME"]
        actors = [Actor.remote(user_config, system_config) for _ in range(num_actor)]
        ray.get([actor.ready.remote(model_path) for actor in actors])

        num_trajectory = validate_config["NUM_TRAJECTORY"]
        pending = []
        for i in range(num_episode):
            actor = actors[i % num_actor]
            pending.append(actor.get_score.remote(num_trajectory))
        
        results = ray.get(pending)
        score = sum(results) / len(results)
        print(f"Result {score}")
        return score >= validate_config["SCORE"]