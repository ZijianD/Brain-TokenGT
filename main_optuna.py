import optuna
from tqdm import tqdm
import torch
from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
from model_transformer import EvolveGCNH_Transformer
from datasets import datasets,label

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score as auc

from config import opt

print(opt)

# Hyperparameter tuning
def objective(trial: optuna.trial.Trial) -> float:
    
    skf = StratifiedKFold(3,shuffle=True,random_state=2023)

    epochs = 50

    keys = np.array(list(datasets.keys()))
    labels = np.array([label[k] for k in keys])
    keys = np.array(list(range(100)))
    labels = np.random.choice([0, 1], size=100)

    depth = trial.suggest_int('depth',opt.K[0],opt.K[-1]) 

    in_channels = datasets.get(keys[0])[0].x.shape[1] ## node feature
    print('in_channels shape:', in_channels)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", opt.lr[0], opt.lr[-1], log=True)

    nhead = trial.suggest_categorical("nhead", opt.n_heads)

    num_layers = trial.suggest_int('num_layers',opt.num_layers[0],opt.num_layers[-1])

    num_grcu_layers= trial.suggest_categorical("num_grcu_layers",opt.num_grcu_layers)

    output_sizes= []
    for i in range(num_grcu_layers):
        if i == num_layers -1:
            output_sizes.append(trial.suggest_int('output_size_'+str(0),10,90))
        else:
            output_sizes.append(trial.suggest_int('output_size_'+str(0),10,90))
    print('output_sizes start here')
    print(output_sizes)
    print(len(output_sizes))
    print('output_sizes end here')
    linear_dim = trial.suggest_categorical("linear_dim",opt.linear_dim)
    
    static_edge_topk = trial.suggest_categorical("static_edge_topk",opt.static_edge_topk)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_aucs = []
    # for i,(train_index,val_test_index) in enumerate(skf.split(keys,labels)):
    train_index=[0,1]; val_index = [2]
    best_auc = 0
    train_g_list = [datasets[k] for k in keys[train_index].tolist()]
    train_g_labels = labels[train_index]
    
    # val_index,test_index = train_test_split(val_index,test_size=0.5,random_state=2023,stratify =labels[val_index])

    # test_g_list = [datasets[k] for k in keys[val_index].tolist()]
    # test_g_labels = labels[val_index]

    val_g_list = [datasets[k] for k in keys[val_index].tolist()]
    val_g_labels = labels[val_index]

    model = EvolveGCNH_Transformer(in_channels,output_sizes,nhead=nhead,num_layers=num_layers,static_edge_topk=static_edge_topk).train()
    # model.to_gpu(device)

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # best_auc=0
          
    for epoch in range(epochs):
        losses = 0.0
        for j,g in enumerate(train_g_list):
            y = torch.as_tensor([train_g_labels[j]])
            A_list = [to_dense_adj(item.edge_index).squeeze(0) for item in g]
            Nodes_list = [item.x.to(torch.float32) for item in g]
            nodes_mask_list = None
            A_list = [i.to(device) for i in A_list]
            Nodes_list = [i.to(device) for i in Nodes_list]
            outputs = model(A_list,Nodes_list,nodes_mask_list)
            loss = F.binary_cross_entropy_with_logits(outputs,y.double().cuda())
            losses+=loss
        optimizer.zero_grad()
        losses = losses/len(train_g_list)
        losses.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            y_preds = []
            for j,g in enumerate(val_g_list):
                A_list = [to_dense_adj(item.edge_index).squeeze(0) for item in g]
                Nodes_list = [item.x.to(torch.float32) for item in g]
                nodes_mask_list = None
                outputs = model(A_list,Nodes_list,nodes_mask_list)
                y_preds.append(torch.sigmoid(outputs).cpu().numpy())
            model.train()
        AUC = auc(val_g_labels,np.nan_to_num(np.hstack(y_preds),0))
        # print(f'epoch {epoch} : val auc = {AUC}')

        if best_auc<AUC:
            best_auc = AUC
    best_aucs.append(best_auc)

    final_auc = np.mean(best_aucs)
    # print('5 folds average aucs =',final_auc)
    # print('*'*500)
    return final_auc

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    #display(study.trials_dataframe().sort_values(by='value',ascending=False).head(1))

    study.trials_dataframe().sort_values(by='value',ascending=False).to_csv('tuning_log.csv',index=False)