import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from torch.nn import functional as F
import utils as u


@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.t()  


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    normalizer = final_matrix.norm(p=2, dim=1, keepdim=True)
    normalizer[normalizer == 0] = 1e-5
    final_matrix = final_matrix / normalizer

    return final_matrix

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation,device = "cpu"):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows)).to(device)
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows)).to(device)
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols)).to(device)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k,device = "cpu"):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1)).to(device)
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):

        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask=None):
        if mask is None:
          mask=0
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask
        
        vals, topk_indices = scores.view(-1).topk(self.k) 
        

        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices,self.k) 
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense() 

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        return out.t()

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

class GRCU(torch.nn.Module):
    def __init__(self,args,device = "cpu"):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats)).to(device)
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):

        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list=None):
        if mask_list is None:
          mask_list = [None]*len(node_embs_list)

        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t]) 

            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq


class EvolveGCNH(torch.nn.Module):
    def __init__(self, in_channels,output_sizes, activation=F.relu, skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [in_channels] + output_sizes

        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)

            self.GRCU_layers.append(grcu_i)
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)
        return out