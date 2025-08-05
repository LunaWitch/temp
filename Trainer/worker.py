from user_define.environment import EnvWrapper
from user_define.model import ModelWrapper

from ray import train
class Worker:
    def __init__(self, user_config, system_config):
        self.config = user_config
        self.system = system_config
        self.learning_rate = self.system["TRAIN"]["LEARNING_RATE"]

        self.model = ModelWrapper(self.config)
        self.env = EnvWrapper(self.config)

    def ready(self, model_path):
        self.model.load_state_dict(model_path)
        model_dict = self.model.get_model()

        for name, model in model_dict.items():
            wrapped = train.torch.prepare_model(model)
            setattr(self.model, name, wrapped)
        self.model.create_optimizer(self.learning_rate)

        for model in self.model.get_model().values():
            model.train()
        return True

    def train_model(self, batch):
        float_keys = ["state", "next_state", "action", "reward", "log_prob", "done"]
        tensors = {k: batch[k].float() for k in float_keys}
        preprocess = {
            k: v.float() for k, v in batch.items() if k not in float_keys
        }

        return self.model.train_model(
                tensors["state"].float(),
                tensors["next_state"].float(),
                tensors["action"].float(),
                tensors["reward"].float(),
                tensors["log_prob"].float(),
                tensors["done"].float(),
                preprocess
            )

    def save_model(self, model_path):
        return self.model.save_model(model_path)
