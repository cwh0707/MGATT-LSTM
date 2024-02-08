import os
import numpy as np
import scipy.sparse as sp 
import torch
from torch.utils.data import Dataset
import pandas as pd

class AirGraph():
    def __init__(self, args):
        graph_dir = args.graph_dir
        self.A_dist = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'city_distances.npy'))))
        self.A_neighb = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'city_neighbor.npy'))))
        self.A_func = torch.from_numpy(np.float32(np.load(os.path.join(graph_dir, 'city_functional.npy'))))
        self.use_graph = args.graph_use
        self.graph_num = len(self.use_graph)
        self.node_num = self.A_dist.shape[0]
        
        self.fix_weight = args.fix_weight
        if self.fix_weight == True:
            self.fix_weight = self.get_fix_weight()
        
    def get_used_graphs(self):
        graph_list = []
        for name in self.use_graph:
            graph_list.append(self.get_graph(name))
        return graph_list

    # fix_weight
    def get_fix_weight(self):
        return (self.A_dist * 0.3591 + \
               self.A_neighb * 0.3808 + \
               self.A_func * 0.2599) / 3

    def get_graph(self, name):
        if name == 'dist':
            return self.A_dist
        elif name == 'neighb':
            return self.A_neighb
        elif name == 'func':
            return self.A_func
        else:
            raise NotImplementedError


class AirDataset(Dataset):
    def __init__(self, args):
        self.hist_len = args.hist_len
        self.pred_len = args.pred_len
        self.x, self.y = self._process_data()
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def _process_data(self):
        path = './data/pollution/airpollution.csv'
        Airpollution = pd.read_csv(path)
        feature = ["AQI","PM2.5_24h","PM10","PM10_24h","SO2","SO2_24h","NO2","NO2_24h","O3","O3_24h","O3_8h","O3_8h_24h","CO", "CO_24h","PM2.5"]
        Airpollution = Airpollution.loc[:, feature]
        x_len = Airpollution.shape[0]
        all_timestamp = x_len // 57  
        node_num = 57 
        
        sample_x = []
        sample_y = []
        # i is Dividing line
        for i in range(self.hist_len, all_timestamp - self.pred_len):
            x = np.float32(Airpollution.iloc[node_num * (i - self.hist_len) : node_num * i,:])
            y = np.float32(Airpollution.iloc[node_num * i:node_num*(i + self.pred_len), -1])
            x = torch.from_numpy(x).view(x.shape[0]//node_num, -1, x.shape[1])
            y = torch.from_numpy(y).view(y.shape[0]//57, node_num)
            sample_x.append(x)
            sample_y.append(y)
            
        sample_x = torch.stack(sample_x)
        sample_y = torch.stack(sample_y)
        return sample_x, sample_y


# def process_data(hist_len, pred_len):
#     path = './data/pollution/airpollution.csv'
#     Airpollution = pd.read_csv(path)
#     feature = ["AQI","PM2.5_24h","PM10","PM10_24h","SO2","SO2_24h","NO2","NO2_24h","O3","O3_24h","O3_8h","O3_8h_24h","CO", "CO_24h","PM2.5"]
#     Airpollution = Airpollution.loc[:, feature]
#     x_len = Airpollution.shape[0]
#     all_timestamp = x_len // 57  # 总共有多少个时间步
#     node_num = 57 # node num
#     # 先划分有多少个时间步
#     sample_x = []
#     sample_y = []
#     # i 为界线
#     for i in range(hist_len, all_timestamp - pred_len):
#         x = np.array(Airpollution.iloc[node_num * (i - hist_len) : node_num * i,:])
#         y = np.array(Airpollution.iloc[node_num * i:node_num*(i+pred_len), -1])
#         x = torch.from_numpy(x).view(x.shape[0]//node_num, -1, x.shape[1])
#         y = torch.from_numpy(y).view(y.shape[0]//57, node_num, -1)
#         sample_x.append(x)
#         sample_y.append(y)
        
#     sample_x = torch.stack(sample_x)
#     sample_y = torch.stack(sample_y)
    
#     return sample_x, sample_y

    