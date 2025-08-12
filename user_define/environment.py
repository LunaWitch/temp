import torch
import gymnasium as gym

class EnvWrapper:
    def __init__(self, config):
        self.env = gym.make('CartPole-v1')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action.item())
        done = terminated or truncated
        return next_state, reward, done
class NetworkEnv:
    def __init__(self):
        # frequency 할당정보
        # 연결정보
        # 현재 CIR
        # 현재 상태의 reward
        pass