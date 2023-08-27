import numpy as np
import time
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import NNConv, Set2Set
import copy
from collections import OrderedDict


from mixture_of_experts import MoE




class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim))
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(torch.einsum('end,edh->enh', x, self.w1))
        #out = self.act(self.norm(hidden1))
        return out
    
class SqueezeLayer(nn.Module):
    
    def __init__(self, fn=True):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        if self.fn:
            tmp =  torch.unsqueeze(x, dim=1)
            return tmp
        else:
            return torch.squeeze(x[0], dim=1), x[1]

class MoELayer(nn.Module):
    
    def __init__(self, hidden_size, num_experts, dropout):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.MoE = MoE(dim = hidden_size, num_experts = num_experts, experts = Experts(hidden_size, num_experts = num_experts))
        self.s1 = SqueezeLayer()
        self.s2 = SqueezeLayer(fn = False)
        self.act_fn = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        raw = x
        x = self.s1(x)
        x = self.MoE(x)
        x, a_loss = self.s2(x)
        x = raw+x
        x = self.act_fn(self.norm(x))
        x = self.dropout(x)
        return x, a_loss

class Fea_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dense_layers = 4, sparse_layers = 4, num_experts = 6, \
                 dropout = 0.2, output_size =  256):
        super(Fea_Encoder, self).__init__()
        self.input_layer =nn.Sequential( 
                            nn.Linear(input_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(p=0.05)
                            ) 


        self.dense = nn.ModuleList([nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.BatchNorm1d(hidden_size),
                            nn.ReLU(),
                            nn.Dropout(p=dropout)) for i in range(dense_layers)])

        
        self.sparse = nn.ModuleList(
            [nn.Sequential(MoELayer(hidden_size,num_experts,dropout)) for i in range(sparse_layers)])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sparse_layers = sparse_layers
        self.dense_layers = dense_layers
        self._initialize_weights()
        self.act =  nn.ReLU()


        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.input_layer(x)
        loss = 0
        for i in range(self.sparse_layers):
            x, tmp_loss = self.sparse[i](x)
            loss += tmp_loss
        for i in range(self.dense_layers):
            x = self.dense[i](x)
        x = self.output_layer(x)
        return x, loss


class CLME(nn.Module):
    def __init__(self, 
                 node_in_feats, edge_in_feats, g_hidden_feats = 64, num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024, s_hidden_feats = 512, s_layers = 4, s_heads = 2, s_output_feats = 256
                 ):
        super(CLME, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats,  hidden_feats = g_hidden_feats, num_step_message_passing = num_step_message_passing, num_step_set2set = num_step_set2set, num_layer_set2set = num_layer_set2set,
                 readout_feats = readout_feats)
        self.transformer = Smile_Encoder(width = s_hidden_feats, layers = s_layers, heads = s_heads, vocab_size = 599,context_length = 320,output_dim = s_output_feats)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gproj = nn.Linear(readout_feats*2,256)
        self.sproj = nn.Linear(s_output_feats,256)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)

    def forward(self, rmols, pmols, smiles):
        smiles_features = self.transformer(smiles)

        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)
        graph_features = torch.cat([r_graph_feats, p_graph_feats], 1)

        graph_features = self.gproj(graph_features)
        smiles_features = self.sproj(smiles_features)

        smiles_features = smiles_features / smiles_features.norm(dim=1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * smiles_features @ graph_features.t()
        logits_per_smiles = logits_per_graph.t()

        return logits_per_graph, logits_per_smiles

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class Smile_Encoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, vocab_size: int, context_length, output_dim ):
        super().__init__()
        self.vocab_size = vocab_size
        self.transformer_width = width
        self.context_length = context_length
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.ln_final = nn.LayerNorm(self.transformer_width)
        self.pooler = nn.Linear(self.transformer_width, output_dim)
        self.transformer = Transformer(width, layers, heads)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        
    def forward(self, text):
        '''
        text: [batch_size: N, context_lenth: L]
        '''
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = self.pooler(x[:,0,:])
        return x   # ND

class FALayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(FALayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['attr'], edges.src['attr']], dim=1)
        t = torch.tanh(self.gate(h2)).squeeze()
        dst_degree = edges.dst['d']
        src_degree = edges.src['d']
        dst_degree = dst_degree.squeeze(-1).expand_as(t)
        src_degree = src_degree.squeeze(-1).expand_as(t)
        e = 0.3 + t * dst_degree * src_degree
        e = self.dropout(e.unsqueeze(1))
        return {'e': e, 'm': t}

    def forward(self, g, node_feats):
        g = copy.deepcopy(g)
        g.ndata['d'] = g.in_degrees().float().unsqueeze(1)
        g.ndata['attr'] = node_feats
        g.apply_edges(self.edge_applying)
        return g.edata['e']  #


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.falayer = FALayer(in_dim=hidden_feats, dropout=0.5)
 
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats+1, hidden_feats * hidden_feats)
        
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
        new_edge_feats = self.falayer(g, node_feats)

        edge_feats = torch.cat([edge_feats, new_edge_feats], dim=-1)
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

class MMModel(nn.Module):

    def __init__(self, 
        mlp_input_size, node_in_feats, edge_in_feats, 
        mlp_hidden_size = 1024, dense_l = 4, spar_l = 2, num_exps = 4, mlp_drop=0.2, mlp_out_size = 256,
        g_hidden_feats = 64, num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1, readout_feats = 1024, 
        s_hidden_feats = 512, s_layers = 4, s_heads = 2, s_output_feats = 256,
        predict_hidden_feats = 512, prob_dropout = 0.2, pre_trained = None):
        
        super(MMModel, self).__init__()
        

        self.mlp = Fea_Encoder(input_size =mlp_input_size , hidden_size=mlp_hidden_size, dense_layers = dense_l, sparse_layers = spar_l, num_experts = num_exps, \
                 dropout = mlp_drop, output_size =  mlp_out_size)
        
        self.clme = CLME(node_in_feats, edge_in_feats,g_hidden_feats = g_hidden_feats, num_step_message_passing = num_step_message_passing, num_step_set2set = num_step_set2set, num_layer_set2set = num_layer_set2set,
        readout_feats = readout_feats, s_hidden_feats = s_hidden_feats, s_layers = s_layers, s_heads = s_heads, s_output_feats = s_output_feats)
        if pre_trained:
            self.clme.load_state_dict(torch.load(pre_trained))


        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2 + mlp_out_size +s_output_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2)
        )
    
    def forward(self, rmols, pmols, input_feats, smiles):

        r_graph_feats = torch.sum(torch.stack([self.clme.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = self.clme.mpnn(pmols) 
        feats, a_loss = self.mlp(input_feats)
        seq_feats = self.clme.transformer(smiles)
        concat_feats = torch.cat([r_graph_feats, p_graph_feats, feats, seq_feats], 1)
        out = self.predict(concat_feats)

        return out[:,0], out[:,1], a_loss