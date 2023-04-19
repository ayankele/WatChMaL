import torch
import torch.nn.functional as F
import torch.nn as nn

# pyg imports
import torch_geometric

class ResGCNEmbed(nn.Module):
    def __init__(self,  in_feat=8, h_feat=128, num_classes=4, num_layers=6, dropout=0.1):
        '''
        Residual Graph Convolutional Network (ResGCN)
        The skip connection operations from the 
        “DeepGCNs: Can GCNs Go as Deep as CNNs?” 
        and “All You Need to Train Deeper GCNs” papers.
        The implemented skip connections includes the pre-activation residual connection ("res+"), 
        the residual connection ("res"), the dense connection ("dense") and no connections ("plain").
        '''
        super().__init__()

        self.node_encoder = nn.Linear(in_feat, h_feat)

        self.layers = nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = torch_geometric.nn.GENConv(
                h_feat, h_feat, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(h_feat, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = torch_geometric.nn.DeepGCNLayer(
                conv, norm, act, block='res+', dropout=dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        #return x
        x = torch_geometric.nn.global_add_pool(x, batch)
        return x

class VariableEmbed(nn.Module):
    def __init__(self, num_var=19, embed_dim=64):
        super().__init__()
        
        self.layer1 = nn.Linear(num_var, int(embed_dim/4))
        self.layer2 = nn.Linear(int(embed_dim/4), int(embed_dim/2))
        self.layer3 = nn.Linear(int(embed_dim/2), embed_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x=self.layer1(x)
        x=self.activation(x)
        x=self.layer2(x)
        x=self.activation(x)
        x=self.layer3(x)
        
        return x
        
class CentralTransformer(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=128, num_encoder_layers = 6, num_heads=8, dropout=0.0, activation='gelu', hidden_dim_factor = 1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.hidden_layer = nn.Linear(hidden_dim_factor * hidden_dim, num_classes)
        
    def forward(self, embed):
        hidden=self.encoder(embed)
        return self.hidden_layer(hidden)

class HybridTransformerNetwork(nn.Module):
    def __init__(self, hit_in_feat=5, var_in_feat=19, num_classes=2, hit_embed_dim=64, var_embed_dim=64, num_heads=8, num_encoder_layers=6):
        super().__init__()
        
        self.hit_embed=ResGCNEmbed(in_feat=hit_in_feat, h_feat=hit_embed_dim)
        self.var_embed=VariableEmbed(num_var=var_in_feat, embed_dim=var_embed_dim)
        self.embed_dim = hit_embed_dim + var_embed_dim
        
        self.transformer=CentralTransformer(num_classes=num_classes, hidden_dim=self.embed_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers)
        
    def forward(self, graph, var_data):
        hit_embed=self.hit_embed(graph)
        var_embed=self.var_embed(var_data)

        x=self.transformer(torch.cat((hit_embed,var_embed),-1).reshape(-1,1,self.embed_dim))
        x=x.flatten(start_dim=1,end_dim=2)

        return x

class CentralLinear(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=128):
        super().__init__()
        
        self.layer1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.layer2 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.layer3 = nn.Linear(int(hidden_dim/4), num_classes)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x=self.layer1(x)
        x=self.activation(x)
        x=self.layer2(x)
        x=self.activation(x)
        x=self.layer3(x)
        
        return x


class HybridLinearNetwork(nn.Module):
    def __init__(self, hit_in_feat=5, var_in_feat=19, num_classes=2, hit_embed_dim=64, var_embed_dim=64, num_heads=8, num_encoder_layers=6):
        super().__init__()

        self.hit_embed=ResGCNEmbed(in_feat=hit_in_feat, h_feat=hit_embed_dim)
        self.var_embed=VariableEmbed(num_var=var_in_feat, embed_dim=var_embed_dim)
        self.embed_dim = hit_embed_dim + var_embed_dim

        self.linearnetwork=CentralLinear(num_classes=num_classes, hidden_dim=self.embed_dim)

    def forward(self, graph, var_data):
        hit_embed=self.hit_embed(graph)
        var_embed=self.var_embed(var_data)

        x=self.linearnetwork(torch.cat((hit_embed,var_embed),-1).reshape(-1,1,self.embed_dim))
        x=x.flatten(start_dim=1,end_dim=2)

        return x
