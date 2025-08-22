# model.py
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import InMemoryDataset, Batch, Data

class ModelWrapper:
    def __init__(self, device, config):
        self.config = config
        self.device = device

        self.actor = Actor(self.config, self.device)
        self.critic = Critic(self.config, self.device)

    def create_optimizer(self, learning_rate):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_model(self):
        return {'actor': self.actor, 'critic': self.critic}

    def save_model(self, model_path):
        actor = getattr(self.actor, 'module', self.actor)
        critic = getattr(self.critic, 'module', self.critic)
        torch.save(
            {
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
            },
            model_path,
        )

    def load_model(self, model_path):
        try:
            model_state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.actor.load_state_dict(model_state_dict['actor'])
            self.critic.load_state_dict(model_state_dict['critic'])
            print(f'model loaded successfully at {model_path}.')
        except FileNotFoundError:
            print(f'model file not found at {model_path}.')
    
    def get_action(self, state, info):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            node_power_attn = info['x']
            edge_power_attn = info['edge_attr']
            edge_index = info['edge_index']
            ptr = info['ptr']
            dist = self.actor(freq_alloc=s, node_power_attn=node_power_attn, edge_power_attn=edge_power_attn, edge_index=edge_index, ptr=ptr)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.reshape(-1), log_prob

    def get_value(self, state, info):
        with torch.no_grad():
            node_power_attn = info['x']
            edge_power_attn = info['edge_attr']
            edge_index = info['edge_index']
            batch = info['batch']
            value = self.critic(freq_alloc=state, node_power_attn=node_power_attn, edge_power_attn=edge_power_attn, edge_index=edge_index, batch=batch)
            return value.item()

    def preprocess_data(self, states, next_states, values, next_values, actions, log_probs, rewards, ongoings, infos):
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        delta = rewards + self.config['GAMMA'] * next_values * ongoings - values
        gae = 0.0
        for i in reversed(range(len(rewards))):
            gae = (delta[i] + self.config['GAMMA'] * self.config['LAMBDA'] * ongoings[i] * gae)
            advantages[i] = gae
        td_target = advantages + values
        if advantages.shape[0] > 1:
            std = advantages.std() + 1e-8
        else:
            std = 1.0
        normalized_advantage = (advantages - advantages.mean()) / std

        return {'advantage': normalized_advantage, 'td_target': td_target}

    def collate_fn(self, batch):
        out = defaultdict(list)
        for k, v in batch.items():
            out[k].append(torch.from_numpy(np.array(v, copy=True)))
        for i in range(len(batch['x'])):
            data = Data(
                x=torch.tensor(batch['x'][i], dtype=torch.float32),
                edge_index=torch.tensor(batch['edge_index'][i], dtype=torch.long),
                edge_attr=torch.tensor(batch['edge_attr'][i], dtype=torch.float32)
            )
            out['graph'].append(data)
        batch_graph = Batch.from_data_list(out['graph']).to(self.device)
        batch_state = torch.cat(out['state'], dim=0).reshape(-1, out['state'][0].shape[-1]).to(self.device)
        batch_value = torch.cat(out['value'], dim=0).to(self.device)
        batch_action = torch.cat(out['action'], dim=0).to(self.device)
        batch_log_prob = torch.cat(out['log_prob'], dim=0).to(self.device)
        batch_advantage = torch.cat(out['advantage'], dim=0).to(self.device)
        batch_td_target = torch.cat(out['td_target'], dim=0).to(self.device)
        return batch_state, batch_value, batch_action, batch_log_prob, batch_advantage, batch_td_target, batch_graph 

    def train_model(self, batch):
        batch_state, batch_value, batch_action, batch_log_prob, batch_advantage, batch_td_target, batch_graph = self.collate_fn(batch)
        batch_pi_dist = self.actor(freq_alloc=batch_state, node_power_attn=batch_graph['x'], edge_power_attn=batch_graph['edge_attr'], edge_index=batch_graph['edge_index'], ptr=batch_graph['ptr'])
        batch_new_prob = batch_pi_dist.log_prob(batch_action)
        ratio = torch.exp(batch_new_prob - batch_log_prob)
        policy_gradient = ratio * batch_advantage
        clipped = torch.clamp(ratio, 1 - self.config['EPS_CLIP'], 1 + self.config['EPS_CLIP'])* batch_advantage
        actor_loss = -torch.min(policy_gradient, clipped).mean()
        batch_value = self.critic(freq_alloc=batch_state,  node_power_attn=batch_graph['x'], edge_power_attn=batch_graph['edge_attr'], edge_index=batch_graph['edge_index'], batch=batch_graph['batch'])
        critic_loss = F.mse_loss(batch_td_target, batch_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        before = next(self.actor.parameters()).clone()
        self.actor_optimizer.step()
        after = next(self.actor.parameters()).clone()
    
        return {
            'loss': actor_loss + critic_loss,
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
        }

class Actor(nn.Module):
    def __init__(self, config, device):
        super(Actor, self).__init__()
        self._device = device
        self._num_freq_ch = config['NUM_FREQ']
        self._power_attn_num_level = config['NUM_POWER_CLASS']
        self._d_model = config['D_MODEL']
        self._n_head = config['N_HEAD']
        self._dim_feedforward = config['DIM_FEEDFORWARD']
        self._num_layers = config['ACTOR_NUM_LAYERS']
        self._dropout = config['DROPOUT']
        self._graph_transformer = GraphTransformer(input_dim=self._num_freq_ch, embedding_dim=self._power_attn_num_level,
                                                   num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
                                                   edge_dim=self._power_attn_num_level,
                                                   dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                                   activation="relu", device=self._device)
        self._output_linear = Linear(in_features=self._d_model, out_features=self._num_freq_ch, bias=True, device=self._device)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._output_linear.weight)
        constant_(self._output_linear.bias, 0.)

    def forward(self, freq_alloc, node_power_attn, edge_power_attn, edge_index, ptr):
        x = self._graph_transformer(input=freq_alloc, embedding=node_power_attn, edge_attr=edge_power_attn,
                                    edge_index=edge_index)
        # get action probability
        logit = self._output_linear(x)
        unallocated_node = (torch.sum(freq_alloc, dim=1, keepdim=True) < 1.0)
        logit = torch.where(condition=unallocated_node, input=logit, other=-torch.inf)

        act_dist = ActDist(logit, ptr, device=self._device)
        return act_dist


class Critic(nn.Module):
    def __init__(self, config, device):
        super(Critic, self).__init__()
        self._device = device
        self._num_freq_ch = config['NUM_FREQ']
        self._power_attn_num_level = config['NUM_POWER_CLASS']
        self._d_model = config['D_MODEL']
        self._n_head = config['N_HEAD']
        self._dim_feedforward = config['DIM_FEEDFORWARD']
        self._num_layers = config['CRITIC_NUM_LAYERS']
        self._dropout = config['DROPOUT']
        self._graph_transformer = GraphTransformer(input_dim=self._num_freq_ch, embedding_dim=self._power_attn_num_level,
                                                   num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
                                                   edge_dim=self._power_attn_num_level,
                                                   dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                                   activation="relu", device=self._device)
        self._output_linear = Linear(in_features=self._d_model, out_features=1, bias=True, device=self._device)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._output_linear.weight)
        constant_(self._output_linear.bias, 0.)

    def forward(self, freq_alloc, node_power_attn, edge_power_attn, edge_index, batch):
        x = self._graph_transformer(input=freq_alloc, embedding=node_power_attn, edge_attr=edge_power_attn,
                                    edge_index=edge_index)
        value = global_mean_pool(x=x, batch=batch)
        value = self._output_linear(value)[:, 0]
        return value


class ActDist:
    def __init__(self, logit, ptr, device):
        self._device = device
        self._ptr = ptr
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_freq_ch = logit.shape[1]
        self._dist_list = []
        for idx in range(self._batch_size):
            l = logit[ptr[idx]: ptr[idx+1], :].to(self._device)
            l = torch.flatten(l)
            if torch.all(torch.isinf(l)):
                dist = None
            else:
                dist = Categorical(logits=l)
            self._dist_list.append(dist)

    def sample(self):
        action = []
        for dist in self._dist_list:
            if dist is not None:
                idx = int(dist.sample())
                node = idx // self._num_freq_ch
                freq = idx % self._num_freq_ch
            else:
                node, freq = -1, -1
            action.append([node, freq])
        action = torch.Tensor(action).to(torch.int).to(self._device)
        return action

    def entropy(self):
        entropy = []
        for dist in self._dist_list:
            entropy.append(dist.entropy())
        entropy = torch.Tensor(entropy).to(self._device)
        return entropy

    def log_prob(self, action):  # action: (batch, 2(node, freq))
        action = torch.as_tensor(action, device=self._device, dtype=torch.long)
        lp = []
        for a, dist in zip(action, self._dist_list):
            if dist is not None:
                node, freq = a[0].item(), a[1].item()
                idx = node * self._num_freq_ch + freq
                lp.append(dist.log_prob(torch.tensor(idx, device=self._device)))
            else:
                lp.append(torch.tensor(-torch.inf, device=self._device))
        lp = torch.stack(lp, dim=0)
        return lp


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformer, self).__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        self._input_linear = Linear(in_features=self._input_dim, out_features=self._d_model, bias=True, device=device)
        self._embedding_linear = Linear(in_features=self._embedding_dim, out_features=self._d_model, bias=True, device=device)
        self._layer_list = nn.ModuleList()
        for _ in range(self._num_layers):
            layer = GraphTransformerLayer(d_model=self._d_model, n_head=self._n_head,
                                          edge_dim=self._edge_dim,
                                          dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                          activation=self._activation, device=self._device)
            self._layer_list.append(layer)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._input_linear.weight)
        xavier_uniform_(self._embedding_linear.weight)
        constant_(self._input_linear.bias, 0.)
        constant_(self._embedding_linear.bias, 0.)

    def forward(self, input, embedding, edge_attr, edge_index):
        input = self._input_linear(input)
        x = self._embedding_linear(embedding)
        for layer in self._layer_list:
            x = x + input
            x = layer(x, edge_attr, edge_index)
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformerLayer, self).__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        # Transformer convolution
        out_channel = d_model // n_head
        self._trans_conv = TransformerConv(in_channels=d_model, out_channels=out_channel, heads=n_head,
                                           concat=True, beta=False, dropout=dropout, edge_dim=edge_dim,
                                           bias=True, root_weight=True).to(device)
        # Feedforward neural network
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True, device=device)
        self.ffnn_dropout = Dropout(dropout)
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True, device=device)
        # Layer norm and dropout
        layer_norm_eps = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # Activation
        self.activation = self._get_activation_fn(activation)
        # Reset parameters
        self._reset_parameters()

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

    def _reset_parameters(self):
        xavier_uniform_(self.ffnn_linear1.weight)
        xavier_uniform_(self.ffnn_linear2.weight)
        constant_(self.ffnn_linear1.bias, 0.)
        constant_(self.ffnn_linear2.bias, 0.)
        self._trans_conv.reset_parameters()

    def forward(self, x, edge_attr, edge_index):
        x2 = self._trans_conv(x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
