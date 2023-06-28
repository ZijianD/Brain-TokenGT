import argparse

parser = argparse.ArgumentParser(description="trainer")

parser.add_argument("--K", nargs='+', type=int,default=[1,2,3,4] ,help="depth")

parser.add_argument("--lr",nargs='+', default=[1e-5,1e-1], type=float, help="learning rate,min and max")

parser.add_argument(
    "--n_heads",nargs='+', default=[1,2,4], type=int, help="number of heads of self-attention"
)

parser.add_argument(
    "--num_layers",nargs='+', default=[1,2,3,4], type=int, help="number of layers of transformer encoder"
)

parser.add_argument(
    "--num_grcu_layers",nargs='+', default=[1,2,3,4], type=int, help="number of layers of GRCU_layers"
)

parser.add_argument("--E", nargs='+',default=[90,180,270], type=int, help="static edge topk")

parser.add_argument(
    "--node_dim",nargs='+',default=[64,128,256,512],
    help="dim of node embedding,edge embedding,type embedding",
)

parser.add_argument(
    "--linear_dim", nargs='+',default=[64,128,256,512],
    help="dim of linear transformation",
)
parser.add_argument(
    "--static_edge_topk",nargs="+",default=[90,180,270],
    help="static_edge_topk"
)
opt = parser.parse_args()
print(opt)
print(opt.lr)