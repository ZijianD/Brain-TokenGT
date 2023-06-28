import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric
from model_grcu import GRCU,gaussian_orthogonal_random_matrix
import utils as u
import numpy as np

class args(object):
  temporal_edge_weights=1 
  max_num_nodes = 270 
  time_steps=3 
  in_channels=90 

def time_alignment(edge_weight=1,max_num_nodes=270,time_steps=3):
  new_adj = torch.zeros((max_num_nodes,max_num_nodes))
  indexs = []
  for i in range(max_num_nodes//time_steps):

    idx = list(range(i,max_num_nodes,max_num_nodes//time_steps))
    for j in range(len(idx)-1):
      indexs.append([idx[j],idx[j+1]])

  for index in indexs:
    left = index[0]
    right = index[-1]
    new_adj[left][right] = edge_weight
  return new_adj,indexs



def DHT(edge_index, batch, add_loops=False,temporal_edge=None):
    device = edge_index.device
    
    temporal_edge = temporal_edge + torch.transpose(temporal_edge,0,1)

    static_edge_index = torch.vstack(torch.where(edge_index!=0)).contiguous()
    temporal_edge_index = torch.vstack(torch.where(temporal_edge!=0)).contiguous()
    temporal_edge_num  = temporal_edge_index.shape[1]//2

    edge_index = torch.hstack([static_edge_index,temporal_edge_index])
    num_edge = edge_index.size(1)

    edge_to_node_index = torch.arange(0,num_edge,1).repeat_interleave(2).view(1,-1).to(device)
    hyperedge_index = edge_index.T.reshape(1,-1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long().to(device)

    edge_batch = hyperedge_index[1,:].reshape(-1,2) 
    edge_batch = edge_batch[:,0] 
    edge_batch = torch.index_select(batch, 0, edge_batch)

    if add_loops:
        bincount =  hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]]!=1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0,num_edge,1).view(1,-1),
                            torch.arange(max_edge+1,max_edge+num_edge+1,1).view(1,-1)],
                            dim=0)
        hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

    return hyperedge_index, edge_batch ,temporal_edge_num

class EvolveGCNH_Transformer(torch.nn.Module):
    def __init__(self, in_channels,output_sizes, 
                 activation=F.relu,nhead=4,num_layers=2,
                 edge_input_channels=1,num_nodes=90,
                 total_graph_size=1,static_edge_topk = 180,device = 'cpu'):
      
        super().__init__()
        GRCU_args = u.Namespace({})

        self.temporal_edge,self.adj_tmp = time_alignment(args.temporal_edge_weights,args.max_num_nodes,args.time_steps)
        self.device = device
        self.nhead = nhead
        self.nhead =1
        self.output_sizes = output_sizes
        self.in_channels = in_channels
        feats = [in_channels] + output_sizes 
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})
            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i)
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))
            
        last_size = output_sizes[-1]

        self.linear = nn.Linear(last_size,last_size) #721
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=last_size, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._parameters.extend(list(self.transformer_encoder.parameters()))
        self.classifier = nn.Linear(last_size,1)
        self._parameters.extend(list(self.classifier.parameters()))

        self.PoolingConvs = pyg_nn.HypergraphConv(edge_input_channels, last_size)
        self._parameters.extend(list(self.PoolingConvs.parameters()))
        self.type_embedding = nn.Embedding(num_embeddings=3,embedding_dim=last_size)
        self._parameters.extend(list(self.type_embedding.parameters()))
        self.static_edge_topk = torch_geometric.nn.pool.TopKPooling(in_channels=last_size,ratio=static_edge_topk)
        self._parameters.extend(list(self.static_edge_topk.parameters()))

        self.projection = nn.Linear(in_channels,256)
        self._parameters.extend(list(self.projection.parameters()))
        
        self.orthogonal_matrix = gaussian_orthogonal_random_matrix(num_nodes,last_size)
        self.graph_token = nn.Embedding(num_embeddings=total_graph_size,embedding_dim=last_size)


    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list,edge_attr=None,graph_id=0,use_node_identifier=True,use_type_identifier=True):
        
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        node_embedding = torch.vstack(Nodes_list)
        
        adjs = torch.block_diag(*A_list) 
        
        adjs_90 = torch.tensor(np.eye(270,270,90),dtype=torch.float32)
        adjs_m90 = torch.tensor(np.eye(270,270,-90),dtype=torch.float32)
        adjs_all = adjs+adjs_90+adjs_m90 
        adjs_all = adjs_all.float()
       
        if edge_attr is None:
          row,col= torch.where(adjs_all!=0)
          edge_attr = adjs_all[row,col]

        batch = torch.zeros(adjs.shape[0]) 

        hyperedge_index, edge_batch,temporal_edge_num = DHT(adjs, batch,temporal_edge=self.temporal_edge)
        hyperedge_index = hyperedge_index.to(self.device)
        
        edge_embedding = F.mish(self.PoolingConvs(edge_attr.view(-1,1).to(self.device), hyperedge_index))
        
        static_edge_embedding = edge_embedding[0:edge_embedding.shape[0]-temporal_edge_num]

        static_edge_index = hyperedge_index[:,0:edge_embedding.shape[0]-temporal_edge_num]
        static_edge_embedding = self.static_edge_topk(static_edge_embedding,static_edge_index)[0]

        temporal_edge_embedding = edge_embedding[edge_embedding.shape[0]-temporal_edge_num:]

        node_type_embedding= self.type_embedding(torch.zeros(node_embedding.shape[0]).long().to(self.device))
        static_edge_type_embedding = self.type_embedding(torch.ones(static_edge_embedding.shape[0]).long().to(self.device))
        temporal_edge_type_embedding = self.type_embedding(2*torch.ones(temporal_edge_num).long().to(self.device))
        if use_type_identifier:
          node_embeddings = node_embedding+node_type_embedding
          static_edge_embeddings = static_edge_embedding+static_edge_type_embedding
          temporal_edge_embeddings = temporal_edge_embedding+temporal_edge_type_embedding
        else:
          node_embeddings = node_embedding
          static_edge_embeddings = static_edge_embedding
          temporal_edge_embeddings = temporal_edge_embedding

        graph_embedding = self.graph_token(torch.tensor(graph_id))

        if use_node_identifier:
          
          all_embeddings = torch.vstack([node_embeddings,static_edge_embeddings,temporal_edge_embeddings,self.orthogonal_matrix])
        else:
          all_embeddings = torch.vstack([node_embeddings,static_edge_embeddings,temporal_edge_embeddings])
          
        graph_embedding = graph_embedding.squeeze(0)

        all_embeddings = torch.vstack([graph_embedding,all_embeddings])

        all_embeddings = self.linear(all_embeddings)

        out = self.transformer_encoder(all_embeddings).mean(0)

        out = self.classifier(out)
        
        return out
    

class TokenGT(torch.nn.Module):
    def __init__(self, in_channels,output_sizes, 
                 activation=F.relu,nhead=4,num_layers=2,
                 edge_input_channels=1,num_nodes=90,
                 total_graph_size=1,static_edge_topk = 180,device = 'cpu'): # skipfeats=False
      
        super().__init__()
        GRCU_args = u.Namespace({})

        self.temporal_edge,self.adj_tmp = time_alignment(args.temporal_edge_weights,args.max_num_nodes,args.time_steps)

        self.device = device
        self.nhead = nhead
        self.nhead =1
        self.output_sizes = output_sizes
        self.in_channels = in_channels

        feats = [in_channels] + output_sizes 

        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i)
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

        last_size = output_sizes[-1]

        self.linear = nn.Linear(last_size,last_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=last_size, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._parameters.extend(list(self.transformer_encoder.parameters()))
        self.classifier = nn.Linear(last_size,1)
        self._parameters.extend(list(self.classifier.parameters()))

        self.PoolingConvs = pyg_nn.HypergraphConv(edge_input_channels, last_size)

        self._parameters.extend(list(self.PoolingConvs.parameters()))


        self.type_embedding = nn.Embedding(num_embeddings=3,embedding_dim=last_size)
        self._parameters.extend(list(self.type_embedding.parameters()))
        
        self.static_edge_topk = torch_geometric.nn.pool.TopKPooling(in_channels=last_size,ratio=static_edge_topk)

        self._parameters.extend(list(self.static_edge_topk.parameters()))

        self.projection = nn.Linear(in_channels,256)
        self._parameters.extend(list(self.projection.parameters()))
        
        self.orthogonal_matrix = gaussian_orthogonal_random_matrix(num_nodes,last_size)
        self.graph_token = nn.Embedding(num_embeddings=total_graph_size,embedding_dim=last_size)


    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list,edge_attr=None,graph_id=0,use_node_identifier=True,use_type_identifier=True):
        
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        node_embedding = torch.vstack(Nodes_list)
        node_embedding = F.mish(node_embedding)
       
        adjs = torch.block_diag(*A_list) 
        
        adjs_90 = torch.tensor(np.eye(270,270,90),dtype=torch.float32)
        adjs_m90 = torch.tensor(np.eye(270,270,-90),dtype=torch.float32)
        adjs_all = adjs+adjs_90+adjs_m90 
        adjs_all = adjs_all.float()
        
        if edge_attr is None:
          row,col= torch.where(adjs_all!=0)
          edge_attr = adjs_all[row,col]


        batch = torch.zeros(adjs.shape[0]) 

        hyperedge_index, edge_batch,temporal_edge_num = DHT(adjs, batch,temporal_edge=self.temporal_edge)
        
        hyperedge_index = hyperedge_index.to(self.device)
        
        edge_embedding = F.mish(self.PoolingConvs(edge_attr.view(-1,1).to(self.device), hyperedge_index))
        
        static_edge_embedding = edge_embedding[0:edge_embedding.shape[0]-temporal_edge_num] 
        static_edge_index = hyperedge_index[:,0:edge_embedding.shape[0]-temporal_edge_num] 
        static_edge_embedding = self.static_edge_topk(static_edge_embedding,static_edge_index)[0] 

        node_type_embedding= self.type_embedding(torch.zeros(node_embedding.shape[0]).long().to(self.device))
        static_edge_type_embedding = self.type_embedding(torch.ones(static_edge_embedding.shape[0]).long().to(self.device))

        if use_type_identifier:
          node_embeddings = node_embedding+node_type_embedding
          static_edge_embeddings = static_edge_embedding+static_edge_type_embedding

        else:
          node_embeddings = node_embedding
          static_edge_embeddings = static_edge_embedding

        graph_embedding = self.graph_token(torch.tensor(graph_id))

        if use_node_identifier:
          
          all_embeddings = torch.vstack([node_embeddings,static_edge_embeddings,self.orthogonal_matrix])
        else:
          all_embeddings = torch.vstack([node_embeddings,static_edge_embeddings,])
          
        graph_embedding = graph_embedding.squeeze(0)

        all_embeddings = torch.vstack([graph_embedding,all_embeddings])

        all_embeddings = self.linear(all_embeddings)

        out = self.transformer_encoder(all_embeddings).mean(0)

        out = self.classifier(out)
        
        return out