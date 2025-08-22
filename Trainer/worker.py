from ray import train
import torch

from util.import_util import get_class
class Worker:
    def __init__(self, model_config, env_config, system_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = system_config['TRAIN']['LEARNING_RATE']
        Model = get_class(model_config['MODEL_MODULE'], model_config['MODEL_CLASS'])
        self.model = Model(self.device, model_config)

    def ready(self, model_path):
        self.model.load_model(model_path)
        model_dict = self.model.get_model()

        for name, model in model_dict.items():
            wrapped = train.torch.prepare_model(model)
            setattr(self.model, name, wrapped)
        self.model.create_optimizer(self.learning_rate)

        for model in self.model.get_model().values():
            model.train()
        return True

    def collate_fn(self, batch):
        return self.model.collate_fn(batch)

    def train_model(self, batch):
        return self.model.train_model(batch)

    def save_model(self, model_path):
        return self.model.save_model(model_path)
