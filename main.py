import ray
import yaml
from Trainer.trainer import Trainer
from util.path_util import get_config_dir, get_user_define_dir

CONFIG_DIR = get_config_dir()
USER_DEFINE_DIR = get_user_define_dir()
TRAIN_YAML_FILE = "system_config.yaml"

def main():
    user_config_name = "user_config.yaml"
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    with open(CONFIG_DIR / TRAIN_YAML_FILE, "r") as f:
        config = yaml.safe_load(f)
        system_config = config

    with open(USER_DEFINE_DIR / user_config_name, "r") as f:
        user_config = yaml.safe_load(f)

    generations = system_config["NUM_GENERATION"]
    trainer = Trainer(system_config)

    for generation in range(generations):
        trainer.run(user_config)
    ray.shutdown()


if __name__ == "__main__":
    main()
