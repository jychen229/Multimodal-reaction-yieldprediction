import torch
import torch.nn as nn
from dgl.nn.pytorch import NNConv, Set2Set

def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1.0)
        nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int, output_size=1):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.ls = nn.ModuleList(
                 [nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.GELU(),
                            nn.Dropout(0.2)) for _ in range(layers)]
        )
        self.l2 = nn.Linear(hidden_size, output_size)
        self.layers = layers
        self.activation = nn.GELU()
        init_params(self.l1)
        init_params(self.l2)

    def forward(self, x):
        x = self.activation(self.l1(x))
        for i in range(self.layers):
            x = self.ls[i](x)
        x = self.l2(x)
        return x


class MPNN(nn.Module):
    def __init__(self, node_in_feats=11, edge_in_feats=3, hidden_feats = 64,
                 num_step_message_passing = 2, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats = 1024,
                 predict_hidden_feats = 512, prob_dropout = 0.1):
        super(reactionMPNN, self).__init__()
        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 1)
        )
    
    def forward(self, x):
        rmols, pmols = x[:2], x[-1]
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = self.mpnn(pmols)
        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)
        return out