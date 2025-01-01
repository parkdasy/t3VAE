import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import *
from train import simulation


parser = argparse.ArgumentParser(description="t3VAE")
parser.add_argument('--dirname',        type=str,   default='results', help='Name of experiments')

parser.add_argument('--nu',             type=float, default=5.0,    help='degree of freedom')
parser.add_argument('--recon_sigma',    type=float, default=0.1,    help='sigma value in decoder')
parser.add_argument('--reg_weight',     type=float, default=1,    help='weight for regularizer term (beta)')

parser.add_argument('--epochs',         type=int,   default=100,    help='Train epoch')
parser.add_argument('--num_layers',     type=int,   default=64,     help='Number of nodes in layers of neural networks')
parser.add_argument('--batch_size',     type=int,   default=1024,   help='Batch size')
parser.add_argument('--lr',             type=float, default=1e-3,   help='Learning rate')
parser.add_argument('--eps',            type=float, default=1e-8,   help="Epsilon for Adam optimizer")
parser.add_argument('--weight_decay',   type=float, default=1e-4,   help='Weight decay')

parser.add_argument('--train_data_seed',type=int,   default=1,      help="Seed for sampling train data")
parser.add_argument('--validation_data_seed', type=int, default=2,  help="Seed for sampling test data")
parser.add_argument('--test_data_seed', type=int,   default=3,      help="Seed for sampling test data")
parser.add_argument('--model_init_seed',type=int,   default=42,     help="Seed for model parameter initialization")

parser.add_argument('--train_ratio',        type=int,   default=0.7 , help="Ratio of sample size of train data")
parser.add_argument('--val_ratio',          type=int,   default=0.15, help="Ratio of sample size of train data")
parser.add_argument('--test_ratio',         type=int,   default=0.15, help="Ratio of sample size of train data")

parser.add_argument('--boot_iter',      type=int,   default=999,    help="Number of iterations in bootstrap MMD test")
parser.add_argument('--gen_N',          type=int,   default=500000, help="Number of generations")
parser.add_argument('--MMD_test_N',     type=int,   default=100000, help="Number of generations")
parser.add_argument('--xlim',           type=float, default=0.2,   help="Maximum value of x-axis in log-scale plot")
parser.add_argument('--patience',       type=int,   default=15,     help="Patience for Early stopping")

args = parser.parse_args()

## from util
def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


n_dim = m_dim = 1

device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
dirname = f'finance/{args.dirname}'

make_reproducibility(args.model_init_seed)
model_list = [
    TVAE.TVAE(device = device).to(device), 
    VAE_st.VAE_st(nu = 9.0, recon_sigma=args.recon_sigma, device=device).to(device),
    VAE_st.VAE_st(nu = 12.0, recon_sigma=args.recon_sigma, device=device).to(device),
    VAE_st.VAE_st(nu = 15.0, recon_sigma=args.recon_sigma, device=device).to(device),
    VAE_st.VAE_st(nu = 18.0, recon_sigma=args.recon_sigma, device=device).to(device),
    VAE_st.VAE_st(nu = 21.0, recon_sigma=args.recon_sigma, device=device).to(device),
]

simulation(
    model_list, [model.model_name for model in model_list], 
    args.train_ratio, args.val_ratio, args.test_ratio,
    dirname, device, args.xlim, 
    args.epochs, args.batch_size, args.lr, args.eps, args.weight_decay, 
    args.train_data_seed, args.validation_data_seed, args.test_data_seed, 
    bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience, 
    exp_number=1
)

make_reproducibility(args.model_init_seed)
model_list = [
    Disentangled_VAE.Disentangled_VAE(nu = 9.0, recon_sigma=args.recon_sigma, device = device, sample_size_for_integral=1).to(device), 
    Disentangled_VAE.Disentangled_VAE(nu = 12.0, recon_sigma=args.recon_sigma, device = device, sample_size_for_integral=1).to(device), 
    Disentangled_VAE.Disentangled_VAE(nu = 15.0, recon_sigma=args.recon_sigma, device = device, sample_size_for_integral=1).to(device), 
    Disentangled_VAE.Disentangled_VAE(nu = 18.0, recon_sigma=args.recon_sigma, device = device, sample_size_for_integral=1).to(device), 
    Disentangled_VAE.Disentangled_VAE(nu = 21.0, recon_sigma=args.recon_sigma, device = device, sample_size_for_integral=1).to(device)
]

simulation(
    model_list, [model.model_name for model in model_list], 
    args.train_ratio, args.val_ratio, args.test_ratio,
    dirname, device, args.xlim, 
    args.epochs, args.batch_size, args.lr, args.eps, args.weight_decay, 
    args.train_data_seed, args.validation_data_seed, args.test_data_seed, 
    bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience, 
    exp_number=2
)

make_reproducibility(args.model_init_seed)
model_list = [
    betaVAE.betaVAE(reg_weight = 0.1, recon_sigma=args.recon_sigma, device=device).to(device), 
    betaVAE.betaVAE(reg_weight = 0.2, recon_sigma=args.recon_sigma, device=device).to(device), 
    betaVAE.betaVAE(reg_weight = 0.5, recon_sigma=args.recon_sigma, device=device).to(device), 
    VAE.VAE(recon_sigma=args.recon_sigma, device=device).to(device), 
    betaVAE.betaVAE(reg_weight = 2.0, recon_sigma=args.recon_sigma, device=device).to(device)
]

simulation(
    model_list, [model.model_name for model in model_list], 
    args.train_ratio, args.val_ratio, args.test_ratio,
    dirname, device, args.xlim, 
    args.epochs, args.batch_size, args.lr, args.eps, args.weight_decay, 
    args.train_data_seed, args.validation_data_seed, args.test_data_seed, 
    bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience, 
    exp_number=3
)

make_reproducibility(args.model_init_seed)
model_list = [
    t3VAE.t3VAE(nu=9.0, recon_sigma=args.recon_sigma, device=device).to(device),
    t3VAE.t3VAE(nu=12.0, recon_sigma=args.recon_sigma, device=device).to(device),
    t3VAE.t3VAE(nu=15.0, recon_sigma=args.recon_sigma, device=device).to(device),
    t3VAE.t3VAE(nu=18.0, recon_sigma=args.recon_sigma, device=device).to(device),
    t3VAE.t3VAE(nu=21.0, recon_sigma=args.recon_sigma, device=device).to(device)
]

simulation(
    model_list, [model.model_name for model in model_list], 
    args.train_ratio, args.val_ratio, args.test_ratio,
    dirname, device, args.xlim, 
    args.epochs, args.batch_size, args.lr, args.eps, args.weight_decay, 
    args.train_data_seed, args.validation_data_seed, args.test_data_seed, 
    bootstrap_iter = args.boot_iter, gen_N = args.gen_N, MMD_test_N = args.MMD_test_N, patience = args.patience, 
    exp_number=4
)

