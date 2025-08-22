import numpy as np
import torch
import ray

from util.import_util import get_class
@ray.remote
class Actor:
    def __init__(self, model_config, env_config, system_config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Model = get_class(model_config['MODEL_MODULE'], model_config['MODEL_CLASS'])
        Env = get_class(env_config['ENV_MODULE'], env_config['ENV_CLASS'])
        self.model = Model(self.device, model_config)
        self.env = Env(self.device, env_config)

    def ready(self, model_path):
        self.model.load_model(model_path)
        for model in self.model.get_model().values():
            model.eval()
        return True

    def _rollout(self, num_episode, num_trajectory):
        print(f'rollout')
        state, info = self.env.reset(num_episode)
        value = self.model.get_value(torch.from_numpy(np.stack(state)).float().to(self.device), info)
        trajectory = []
        score = 0.0
        for i in range(num_trajectory):
            old_state = state.clone()
            action, log_prob = self.model.get_action(state, info)
            next_state, reward, ongoing, info = self.env.step(action)
            next_value = self.model.get_value(torch.from_numpy(np.stack(next_state)).float().to(self.device), info)
            trajectory.append((
                old_state, 
                next_state.clone(), 
                value, 
                next_value, 
                action.clone(), 
                log_prob.clone(), 
                reward, 
                ongoing, 
                info
                ))
            score += reward
            state = next_state
            value = next_value
            
            if ongoing is not True:
                break

        return trajectory, score

    def get_episode(self, num_episode, num_trajectory):
        print(f'get_episode')
        trajectory, score = self._rollout(num_episode, num_trajectory)

        states, next_states, values, next_values, actions, log_probs, rewards, ongoings, infos = zip(*trajectory)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        values = torch.from_numpy(np.stack(values)).float().to(self.device)
        next_values = torch.from_numpy(np.stack(next_values)).float().to(self.device)
        actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
        log_probs = torch.from_numpy(np.stack(log_probs)).float().to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).float().to(self.device)
        ongoings = torch.from_numpy(np.stack(ongoings)).float().to(self.device)
        print(f'{len(rewards)} {score}')
        print(f'{rewards}')
        if infos is not None :
            keys = infos[0].keys()
            infos = {k: [info[k] for info in infos] for k in keys}
        extra_data = self.model.preprocess_data(states, next_states, values, next_values, actions, log_probs, rewards, ongoings, infos)
    
        item_list = []
        for t in range(len(rewards)):
            item = {
                'state': states[t].cpu().numpy(),
                'next_state': next_states[t].cpu().numpy(),
                'value': values[t].cpu().numpy(),
                'next_value': next_values[t].cpu().numpy(),
                'action': actions[t].cpu().numpy(),
                'log_prob': log_probs[t].cpu().numpy(),
                'reward': rewards[t].cpu().numpy(),
                'ongoing': ongoings[t].cpu().numpy(),
            }
            for k, v in infos.items():
                item[k] = v[t]
            for k, v in extra_data.items():
                item[k] = v[t].cpu().numpy()
            item_list.append(item)
        return item_list

    def get_score(self, num_episode, num_trajectory):
        trajectory, score = self._rollout(num_episode, num_trajectory)
        states, next_states, values, next_values, actions, log_probs, rewards, ongoings, infos = zip(*trajectory)
        rewards = torch.from_numpy(np.stack(rewards)).float().to(self.device)
        print(f'{len(rewards)} {score}')
        print(f'{rewards}')
        return score
