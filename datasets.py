import tqdm
import os
import json
import scipy
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

def get_datasets(datasets_path:str , label_path:str):

    mats = os.listdir(datasets_path)
    datasets = {}

    for m in tqdm.tqdm(mats):
        dataset = []
        for i in tqdm.tqdm(range(1,4)):
            new_path = datasets_path+m+r'/t_{}/FC/'.format(i)
            if '.DS_Store' in new_path:
                break
            mat = os.listdir(new_path)
            mat=[item for item in mat if 'mat' in item][0]
            FC = scipy.io.loadmat(new_path+mat)
            keys = list(FC.keys())[-1]
            FC = FC[keys].astype('int64')
            FC = torch.from_numpy(FC)
            fcS = torch.where(torch.abs(FC)>0, 1, 0)


            # S = fcS + mcS
            # denseAdj = torch.where(S>0, 1, 0)
            entries = torch.topk(FC.flatten(), 1216).values
            denseAdj = torch.where(FC>=entries[-1],1,0)
            # denseAdj = torch.where(FC>0.0,FC,0.0)

            x = FC
            edge_index, _ = dense_to_sparse(denseAdj)
            edge_attr = torch.where(FC>=entries[-1], FC, torch.tensor(0).long())
            edge_attr = edge_attr.reshape(-1,1)
            edge_attr = edge_attr[edge_attr!=0]
            edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())


            dataset.append(Data(torch.nan_to_num(x,0,1,1), edge_index, edge_attr=edge_attr))
        if '.DS_Store' in new_path:
            continue
        datasets[m] = dataset


    f = open(label_path)

    label = dict(json.load(f))
    return datasets,label

datasets_path ="synthetic_data/"
label_path = 'synthetic_data.json'
datasets,label = get_datasets(datasets_path,label_path)