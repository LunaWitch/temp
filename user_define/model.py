# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class ModelWrapper:
    def __init__(self, config):
        self.config = config["MODEL"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.config["NUM_STATE"], self.config["NUM_ACTION"]).to(self.device)
        self.critic = Critic(self.config["NUM_STATE"]).to(self.device)

    def create_optimizer(self, learning_rate):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_model(self):
        return {"actor": self.actor, "critic": self.critic}

    def save_model(self, model_path):
        actor = getattr(self.actor, "module", self.actor)
        critic = getattr(self.critic, "module", self.critic)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
            },
            model_path,
        )

    def load_state_dict(self, model_path):
        try:
            model_state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.actor.load_state_dict(model_state_dict["actor"])
            self.critic.load_state_dict(model_state_dict["critic"])
            print(f"Model loaded successfully at {model_path}.")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}.")

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            dist = self.actor.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

    def preprocess_data(self, states, next_states, actions, rewards, log_probs, dones):
        with torch.no_grad():
            values = self.critic(states).view(-1)
            next_values = self.critic(next_states).view(-1)

            advantages = torch.zeros_like(rewards, dtype=torch.float32)
            delta = rewards + self.config["GAMMA"] * next_values * (1 - dones) - values
            gae = 0.0
            for i in reversed(range(len(rewards))):
                gae = (delta[i] + self.config["GAMMA"] * self.config["LAMBDA"] * (1 - dones[i]) * gae)
                advantages[i] = gae

            td_target = advantages + values

            if advantages.shape[0] > 1:
                std = advantages.std() + 1e-8
            else:
                std = 1.0  # 또는 1e-8
            normalized_advantage = (advantages - advantages.mean()) / std

        return {"advantage": normalized_advantage, "td_target": td_target}


    def train_model(self, batch_state, batch_next_state, batch_action, batch_reward, batch_log_prob, batch_done, batch_preprocess):
        batch_advantage = batch_preprocess["advantage"]
        batch_td_target = batch_preprocess["td_target"]
        batch_value = self.critic(batch_state)
        batch_pi_dist = self.actor(batch_state)
        batch_new_prob = batch_pi_dist.log_prob(batch_action)
        ratio = torch.exp(batch_new_prob - batch_log_prob)
        policy_gradient = ratio * batch_advantage
        clipped = torch.clamp(ratio, 1 - self.config["EPS_CLIP"], 1 + self.config["EPS_CLIP"])* batch_advantage
        actor_loss = -torch.min(policy_gradient, clipped).mean()

        critic_loss = F.mse_loss(batch_td_target.view(-1), batch_value.view(-1))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return {
            "loss": actor_loss + critic_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return Categorical(logits=self.fc(x))


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)
