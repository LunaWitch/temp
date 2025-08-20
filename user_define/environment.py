import torch
import gymnasium as gym

class EnvWrapper:
    def __init__(self, device, config):
        self.env = gym.make('CartPole-v1')
        self.device = device

    def reset(self, num_episode):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
class NetworkEnv:
    def __init__(self):
        # frequency 할당정보
        # 연결정보
        # 현재 CIR
        # 현재 상태의 reward
        pass