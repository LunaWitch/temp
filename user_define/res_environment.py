import torch
import pyarrow as pa
from torch_geometric.data import InMemoryDataset, Batch, Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from util.path_util import get_user_define_dir

class EnvWrapper:
    def __init__(self, device, config):
        self.config = config
        self.device = device
        self.env = NetworkEnv(config, self.device)

    def reset(self, num_episode):
        graph, info = self.env.reset(num_episode)
        return graph, info

    def step(self, action):
        next_state, reward, ongoing, info = self.env.step(action)
        return next_state, reward, ongoing, info

class NetworkEnv:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        dataset_name = self.config['DATASET']

        self.power_min = self.config['POWER_MIN']
        self.power_max = self.config['POWER_MAX']
        self.power_class = self.config['NUM_POWER_CLASS']
        self.cir_threshold = self.config['CIR_THRESHOLD']
        self.power_boundaries = torch.linspace(self.power_min, self.power_max, self.power_class).to(self.device)

        self.num_freq = self.config['NUM_FREQ']
        self.graph_dataset = InterfGraphDataset(dataset_name)
        self.graph_dataloader = DataLoader(self.graph_dataset, batch_size=1, shuffle=True)
        self.graph_data_iter = iter(self.graph_dataloader)

    def reset(self, num_episode):
        if num_episode is not len(self.graph_dataloader):
            print(f'Please check -> len(batch) : {len(self.graph_dataloader)} / num_episode : {num_episode}')
        self.power_graph = next(self.graph_data_iter).to(self.device)
        num_node = self.power_graph.batch.shape[0]
        batch_size = self.power_graph.ptr.shape[0] - 1
        self.reward = 0
        self.prev_instant_reward = 0
        self.quantized_power_graph = self.quantize_power_attn(self.power_graph)
        self.freq_alloc = torch.zeros(size=(num_node, self.num_freq)).float().to(self.device)
        self.ongoing = torch.full(size=(batch_size,), fill_value=True).to(self.device)
        self.allocated_node = torch.full(size=(num_node,), fill_value=False).to(self.device)
        user_define = {
            'x': self.quantized_power_graph['x'],
            'edge_attr': self.quantized_power_graph['edge_attr'],
            'edge_index': self.quantized_power_graph['edge_index'],
            'ptr': self.quantized_power_graph['ptr'],
            'batch': self.quantized_power_graph['batch'],
        }
        return self.freq_alloc, user_define # state, info

    def step(self, action):
        node, freq = action[0], action[1]
        self.freq_alloc[node, freq] = 1.0
        self.allocated_node[node] = True
        user_define = {
            'x': self.quantized_power_graph['x'],
            'edge_attr': self.quantized_power_graph['edge_attr'],
            'edge_index': self.quantized_power_graph['edge_index'],
            'ptr': self.quantized_power_graph['ptr'],
            'batch': self.quantized_power_graph['batch'],
        }
        instant_reward = self.cal_reward(self.power_graph, self.freq_alloc)
        self.reward = instant_reward - self.prev_instant_reward
        self.prev_instant_reward = instant_reward

        return self.freq_alloc.clone(), self.reward, not torch.all(self.allocated_node), user_define # next_state, reward, done, info
    def quantize_power_attn(self, graph):
        graphs = graph.to_data_list()
        power_graphs = []
        for graph in graphs:
            node_power = graph.get_tensor('x').to(self.device)
            node_power = torch.bucketize(node_power, self.power_boundaries, right=True) - 1
            node_power[node_power == -1] = 0
            node_power = F.one_hot(node_power, num_classes=self.power_class).to(torch.float32)
            edge_power_attn = graph.get_tensor('edge_attr').to(self.device)
            edge_power_attn = torch.bucketize(edge_power_attn, self.power_boundaries, right=True) - 1
            valid_edge_idx = edge_power_attn >= 0
            edge_power_attn = edge_power_attn[valid_edge_idx]
            edge_power_attn = F.one_hot(edge_power_attn, num_classes=self.power_class).to(torch.float32)
            edge_index = graph.edge_index.to(self.device)
            edge_index = edge_index[:, valid_edge_idx]

            power_graph = Data(x=node_power, edge_index=edge_index, edge_attr=edge_power_attn)
            power_graphs.append(power_graph)
        return Batch.from_data_list(power_graphs)
    
    def cal_reward(self, power_graph, freq_alloc):
        cir = self.cal_cir(power_graph, freq_alloc)
        success = (cir >= self.cir_threshold)
        return torch.sum(success.int())
        
    def cal_cir(self, power_graph, freq_alloc):
        # get tx and rx power
        node_power = power_graph.x[:, None]
        tx_power = power_graph.node_tx_power[:, None]
        tx_power = tx_power + 10 * torch.log10(freq_alloc)
        rx_power = tx_power + node_power
        # get interference
        edge_power_attn = power_graph.edge_attr[:, None]
        num_edge = edge_power_attn.shape[0]
        edge_index = power_graph.edge_index
        index_j, index_i = edge_index[0, :], edge_index[1, :]
        index_j = torch.broadcast_to(index_j[:, None], size=(num_edge, self.num_freq))
        index_i = torch.broadcast_to(index_i[:, None], size=(num_edge, self.num_freq))
        tx_power_j = torch.gather(input=tx_power, dim=0, index=index_j)
        interf_db = tx_power_j + edge_power_attn
        interf = torch.pow(10, interf_db * 0.1)
        num_node = freq_alloc.shape[0]
        sum_interf = torch.zeros(size=(num_node, self.num_freq)).to(self.device)
        sum_interf = torch.scatter_add(input=sum_interf, dim=0, index=index_i, src=interf)
        sum_interf_db = 10 * torch.log10(sum_interf)

        node_freq_unalloc = (freq_alloc < 1)
        rx_power[node_freq_unalloc] = 0.0
        rx_power = torch.sum(rx_power, dim=1)
        sum_interf_db[node_freq_unalloc] = 0.0
        sum_interf_db = torch.sum(sum_interf_db, dim=1)
        cir = rx_power - sum_interf_db
        node_unalloc = torch.all(node_freq_unalloc, dim=1)
        cir[node_unalloc] = -torch.inf
        return cir

class InterfGraphDataset(InMemoryDataset):
    def __init__(self, file_name):
        super().__init__()
        self._file_name = get_user_define_dir() / 'network' / file_name
        self.load(str(self._file_name))
