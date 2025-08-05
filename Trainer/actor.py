import numpy as np
import torch
import ray

from user_define.environment import EnvWrapper
from user_define.model import ModelWrapper
@ray.remote
class Actor:
    def __init__(self, user_config, system_config):
        self.config = user_config
        self.system = system_config

        self.model = ModelWrapper(self.config)
        self.env = EnvWrapper(self.config)

    def ready(self, model_path: str) -> bool:
        self.model.load_state_dict(model_path)
        for model in self.model.get_model().values():
            model.eval()
        return True

    def _rollout(self, num_trajectory: int):
        state = self.env.reset()
        trajectory = []
        score = 0.0

        for _ in range(num_trajectory):
            action, log_prob = self.model.get_action(state)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, next_state, action, reward, log_prob, done))
            score += reward
            state = next_state
            if done:
                break

        return trajectory, score

    def get_episode(self, num_trajectory: int):
        trajectory, _ = self._rollout(num_trajectory)

        states, next_states, actions, rewards, log_probs, dones = zip(*trajectory)

        states = torch.from_numpy(np.stack(states)).float()
        next_states = torch.from_numpy(np.stack(next_states)).float()
        actions = torch.from_numpy(np.stack(actions)).float()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        extra_data = self.model.preprocess_data(states, next_states, actions, rewards, log_probs, dones)

        episode_data = []
        for i in range(len(trajectory)):
            item = {
                "state": states[i],
                "next_state": next_states[i],
                "action": actions[i],
                "reward": rewards[i],
                "log_prob": log_probs[i],
                "done": dones[i],
            }
            for k, v in extra_data.items():
                item[k] = v[i]
            episode_data.append(item)

        return episode_data

    def get_score(self, num_trajectory: int):
        _, score = self._rollout(num_trajectory)
        return score
