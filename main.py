import ray

from Trainer.train import run


def main():
    user_config_name = "user_config.yaml"
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    run(user_config_name)
    ray.shutdown()


if __name__ == "__main__":
    main()
