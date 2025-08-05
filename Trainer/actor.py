import ray

from user_define.environment import EnvWrapper
from user_define.model import ModelWrapper
from util.path_util import get_result_dir

RESULT_DIR = get_result_dir()
@ray.remote
class Actor:
    def __init__(self, model_path, user_config, system_config):
        self.model_path = model_path
        self.config = user_config
        self.system = system_config

        self.model = ModelWrapper(self.model_path, self.config)
        self.env = EnvWrapper(self.config)

        model_dict = self.model.get_model()
        for k in model_dict:
            model_dict[k].eval()

    def ready(self):
        return True
    
    def get_episode(self, num_trajectory):
        state = self.env.reset()
        # 고정 연결상태, Power / 변화 할당상태, CIR
        trajectory = []
        score = 0
        for t in range(num_trajectory):
            # 현재 GRAPH 기준 할당 진행 -> 할당상태 업데이트
            action, log_prob = self.model.get_action(state)

            # 할당 결과 기준으로 reward (terminated truncated 없을 듯)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, next_state, action, reward, log_prob, done))
            # 할당 결과 바탕으로 Update된 GRAPH 연결상태 변화
            state = next_state
            score += reward
            if done:
                break
        return trajectory

    def get_score(self, num_trajectory):
        state = self.env.reset()
        # 고정 연결상태, Power / 변화 할당상태, CIR
        trajectory = []
        score = 0
        for t in range(num_trajectory):
            # 현재 GRAPH 기준 할당 진행 -> 할당상태 업데이트
            action, log_prob = self.model.get_action(state)

            # 할당 결과 기준으로 reward (terminated truncated 없을 듯)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, next_state, action, reward, log_prob, done))
            # 할당 결과 바탕으로 Update된 GRAPH 연결상태 변화
            state = next_state
            score += reward
            if done:
                break
        return score