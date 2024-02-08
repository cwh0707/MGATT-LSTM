import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Sequential, Linear, Sigmoid
from torch_geometric.utils import dense_to_sparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    return: [num_vertices, num_graphs, num_vertices, D]
    D:      feature
    '''
    def __init__(self, M, d, bn_decay):
        super(spatialAttention, self).__init__()
        self.d = d
        self.M = M
        D = self.M * self.d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X):
        num_vertex = X.shape[0] 

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)                # [M * num_vertices, num_graphs, num_vertices, d]

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, num_vertex, dim=0), dim=-1)
        X = self.FC(X)
        return X


class graphAttention(nn.Module):
    '''
    multi-graph attention mechanism
    X:      [num_vertices, num_graphs, num_vertices, D]
    SGE:    [num_vertices, num_graphs, num_vertices, D]
    M:      number of attention heads
    d:      dimension of each attention outputs
    D:      feature
    return: [num_vertices, num_graphs, num_vertices, D]
    '''
    def __init__(self, M, d, bn_decay, mask=True):
        super(graphAttention, self).__init__()
        self.d = d  # the dimension of K
        self.M = M  # nhead
        D = self.M * self.d
        self.mask = mask
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X):
        num_vertex_ = X.shape[0]

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)

        query = torch.cat(torch.split(query, self.M, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.M, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.M, dim=-1), dim=0)    

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)

        if self.mask:
            num_vertex = X.shape[0]
            num_step = X.shape[1]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * num_vertex, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)

        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, num_vertex_, dim=0), dim=-1)
        X = self.FC(X)
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [num_vertices, num_graphs, num_vertices, D]
    HG:     [num_vertices, num_graphs, num_vertices, D]
    D:      output dims = M * d
    return: [num_vertices, num_graphs, num_vertices, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HG):
        XS = self.FC_xs(HS)
        XG = self.FC_xt(HG)
        z = torch.sigmoid(torch.add(XS, XG))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HG))
        H = self.FC_h(H)
        return H

class MGABlock(nn.Module):
    def __init__(self, M, d, bn_decay, mask=False):
        super(MGABlock, self).__init__()
        self.spatialAttention = spatialAttention(M, d, bn_decay)
        self.graphAttention = graphAttention(M, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(M * d, bn_decay)

    def forward(self, X):
        HS = self.spatialAttention(X)
        HT = self.graphAttention(X)
        # fusion gate
        H = self.gatedFusion(HS, HT)
        return torch.add(X, H)
    
class MGAtt(nn.Module):
    def __init__(self, graph, matrix_weight, attention, M, d, bn_decay, feature_dim):
        super(MGAtt, self).__init__()
        self.M = M
        self.d = d
        self.bn_decay = bn_decay
        D = self.M * self.d
        self.MGABlock = MGABlock(M, d, bn_decay)

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=self.bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=self.bn_decay)

        self.graph = graph
       
        self.matrix_w = matrix_weight
      
        self.attention = attention
        
        # aggregation operator
        self.X_linear = nn.Linear(feature_dim, self.graph.node_num)

        if self.graph.graph_num == 1: 
            self.fusion_graph = False
            self.A_single = self.graph.get_graph(graph.use_graph[0])
        else:
            self.fusion_graph = True
            self.softmax = nn.Softmax(dim=1)

            if self.matrix_w:
                adj_w = nn.Parameter(torch.randn(self.graph.graph_num, self.graph.node_num, self.graph.node_num))
                adj_w_bias = nn.Parameter(torch.randn(self.graph.node_num, self.graph.node_num))
                self.adj_w_bias = nn.Parameter(adj_w_bias, requires_grad=True)
                self.linear = linear(3, 1)  

            else:
                adj_w = nn.Parameter(torch.randn(1, self.graph.graph_num))

            self.adj_w = nn.Parameter(adj_w, requires_grad=True)
            self.used_graphs = self.graph.get_used_graphs()
            assert len(self.used_graphs) == self.graph.graph_num

    def forward(self, X):
        # fix weight
        if self.graph.fix_weight:
            return self.graph.get_fix_weight()

        if self.fusion_graph:  
            if not self.matrix_w:    
                self.A_w = self.softmax(self.adj_w)[0] # 
                adj_list = [self.used_graphs[i] * self.A_w[i] for i in range(self.graph.graph_num)]
                self.adj_for_run = torch.sum(torch.stack(adj_list), dim=0)      

            else:  
                if self.attention:
                    W = torch.stack((self.used_graphs)).to(device)  # multi graph stack
                    W = self.FC_1(torch.unsqueeze(W.permute(1, 0, 2), -1))
                    W = self.MGABlock(W)     
                    # multi-graph spatial attention
                    W = self.FC_2(W).squeeze(dim=-1)
                    W = torch.sum(self.adj_w * W.permute(1, 0, 2), dim=0)
                else:
                    W= torch.sum(self.adj_w * torch.stack(self.used_graphs), dim=0)
                act = nn.ReLU()
                W = act(W)
                self.adj_for_run = W

        else:
            self.adj_for_run = self.A_single

        # aggregation layer
        X = self.X_linear(X)
        self.adj_for_run = self.adj_for_run.unsqueeze(0).unsqueeze(0)
        X = X * self.adj_for_run 
             
        return X