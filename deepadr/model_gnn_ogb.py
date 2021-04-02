# source: https://github.com/snap-stanford/ogb/blob/153e37636009cac0aeb388073ab6df9f3b2792bf/examples/graphproppred/mol/gnn.py

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from .conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
                #x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        h_node = self.gnn_node(x, edge_index, edge_attr)

        h_graph = self.pool(h_node, batch)

#         return self.graph_pred_linear(h_graph)
        return h_graph

def _init_model_params(named_parameters):
    for p_name, p in named_parameters:
        param_dim = p.dim()
        if param_dim > 1: # weight matrices
            nn.init.xavier_uniform_(p)
        elif param_dim == 1: # bias parameters
            if p_name.endswith('bias'):
                nn.init.uniform_(p, a=-1.0, b=1.0)

class DeepAdr_SiameseTrf(nn.Module):

    def __init__(self, input_dim, dist, num_classes=2):
        
        super().__init__()
        
        if dist == 'euclidean':
            self.dist = nn.PairwiseDistance(p=2, keepdim=True)
            self.alpha = 0
        elif dist == 'manhattan':
            self.dist = nn.PairwiseDistance(p=1, keepdim=True)
            self.alpha = 0
        elif dist == 'cosine':
            self.dist = nn.CosineSimilarity(dim=1)
            self.alpha = 1

        self.Wy = nn.Linear(2*input_dim+1, num_classes)
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_params_()
        print('updated')
        
        
    def _init_params_(self):
        _init_model_params(self.named_parameters())
    
    def forward(self, Z_a, Z_b):
        """
        Args:
            Z_a: tensor, (batch, embedding dim)
            Z_b: tensor, (batch, embedding dim)
        """

        dist = self.dist(Z_a, Z_b).reshape(-1,1)
        # update dist to distance measure if cosine is chosen
        dist = self.alpha * (1-dist) + (1-self.alpha) * dist
        
        out = torch.cat([Z_a, Z_b, dist], axis=-1)
        y = self.Wy(out)
        return self.log_softmax(y), dist

# if __name__ == '__main__':
#     GNN(num_tasks = 10)