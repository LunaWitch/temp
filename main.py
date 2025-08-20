import ray
import yaml
from Trainer.trainer import Trainer
from util.path_util import get_config_dir, get_user_define_dir

CONFIG_DIR = get_config_dir()
USER_DEFINE_DIR = get_user_define_dir()
TRAIN_YAML_FILE = 'system_config.yaml'

def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    user_config_name = 'res_user_config.yaml'

    with open(CONFIG_DIR / TRAIN_YAML_FILE, 'r') as f:
        config = yaml.safe_load(f)
        system_config = config

    with open(USER_DEFINE_DIR / user_config_name, 'r') as f:
        user_config = yaml.safe_load(f)

    epochs = system_config['EPOCH']
    trainer = Trainer(system_config)

    for _ in range(epochs):
        if trainer.run(user_config):
            break
    ray.shutdown()


if __name__ == '__main__':
    main()
