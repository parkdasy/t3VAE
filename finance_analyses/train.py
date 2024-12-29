import os
import copy
import numpy as np
import FinanceDataReader as fdr

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from mmd import make_masking, mmd_linear, mmd_linear_bootstrap_test


## from util
class TensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    def __len__(self):
        return self.tensors[0].size(0)

## from util
def make_result_dir(dirname):
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(dirname + '/generations', exist_ok=True)

## from sampling
def sample_generation (device, SEED = None, ratio = 0.5) : 
    np.random.seed(SEED)

    spy = fdr.DataReader('SPY', start='2014-01-01', end='2024-01-01')
    close_pct_change = spy['Close'].pct_change().to_numpy()
    n = int(ratio * len(close_pct_change))
    data = np.random.choice(close_pct_change, size = n, replace = False)
    data = torch.tensor(data, dtype=torch.float32)

    return data.to(device)

## from visualize
def visualize_density(model_title_list, model_gen_list, xlim) :
    model_gen_list = [gen[torch.isfinite(gen)].cpu().numpy() for gen in model_gen_list]

    M = len(model_gen_list)
    spy = fdr.DataReader('SPY', start='2014-01-01', end='2024-01-01')
    close_pct_change = spy['Close'].pct_change().to_numpy()

    # plot
    fig = plt.figure(figsize = (3.5 * M, 7))

    for m in range(M) : 
        ax = fig.add_subplot(2,M,m+1)
        plt.hist(close_pct_change, bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='orange')
        plt.hist(model_gen_list[m], bins = 100, range = [-10, 10], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-10, 10)
        plt.title(f'{model_title_list[m]}')

        ax = fig.add_subplot(2,M,M+m+1)
        plt.hist(close_pct_change, bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='orange')
        plt.hist(model_gen_list[m], bins = 100, range = [-xlim, xlim], density=True, alpha = 0.5, color='dodgerblue')
        plt.xlim(-xlim, xlim)
        plt.yscale("log")
        plt.ylim(1e-6, 1)

    return fig










def simulation(
        model_list, model_title_list, 
        train_ratio, val_ratio, test_ratio,
        dir_name, device, xlim, 
        epochs, batch_size, lr, eps, weight_decay, 
        train_data_seed, validation_data_seed, test_data_seed, 
        bootstrap_iter = 1999, gen_N = 100000, MMD_test_N = 100000, patience = 10, exp_number = 1) : 
    M = len(model_list)

    dirname = f'./{dir_name}'
    make_result_dir(dirname)

    generation_writer = SummaryWriter(dirname + '/generations')
    model_writer_list = [SummaryWriter(dirname + f'/{title}') for title in model_title_list]

    # Generate dataset
    train_data = sample_generation(
        device, SEED=train_data_seed, ratio = train_ratio
    )
    validation_data = sample_generation(
        device, SEED=validation_data_seed, ratio = val_ratio
    )
    test_data = sample_generation(
        device, SEED=test_data_seed, ratio = test_ratio
    )

    train_dataset = TensorDataset(train_data)

    # Model training with early stopping
    best_loss_list = [10^6 for _ in range(M)]
    best_model_list = copy.deepcopy(model_list)
    model_count = [0 for _ in range(M)]
    model_stop = [False for _ in range(M)]

    opt_list = [optim.Adam(model.parameters(), lr = lr, eps = eps, weight_decay=weight_decay) for model in model_list]

    train_loader_list = [torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) for _ in range(M)]

    for epoch in tqdm(range(0, epochs)) : 
        # breaking code for early stopping
        if all(model_stop) : 
            break

        # 1 epoch train
        for m in range(M) : 
            if model_stop[m] is not True : 
                model_list[m].train()

                denom_train = int(len(train_dataset)/batch_size) + 1

                # gradient descent
                for batch_idx, data in enumerate(train_loader_list[m]) : 
                    data = data[0].to(device)
                    opt_list[m].zero_grad()
                    recon_loss, reg_loss, train_loss = model_list[m](data) # train
                    train_loss.backward()

                    current_step_train = epoch * denom_train + batch_idx
                    model_writer_list[m].add_scalar("Train/Reconstruction Error", recon_loss.item(), current_step_train)
                    model_writer_list[m].add_scalar("Train/Regularizer", reg_loss.item(), current_step_train)
                    model_writer_list[m].add_scalar("Train/Total Loss" , train_loss.item(), current_step_train)

                    opt_list[m].step()

                # validation step
                model_list[m].eval()
                data = validation_data.to(device)
                recon_loss, reg_loss, validation_loss = model_list[m](data) # validation

                model_writer_list[m].add_scalar("Validation/Reconstruction Error", recon_loss.item(), epoch)
                model_writer_list[m].add_scalar("Validation/Regularizer", reg_loss.item(), epoch)
                model_writer_list[m].add_scalar("Validation/Total Loss" , validation_loss.item(), epoch)

                if validation_loss < best_loss_list[m] : 
                    best_loss_list[m] = validation_loss
                    best_model_list[m] = copy.deepcopy(model_list[m])
                    model_count[m] = 0
                else : 
                    model_count[m] += 1
                
                if model_count[m] == patience : 
                    model_stop[m] = True
                    print(f"{model_title_list[m]} stopped training at {epoch-patience}th epoch")

                # test step
                best_model_list[m].eval()
                data = test_data.to(device)
                recon_loss, reg_loss, test_loss = best_model_list[m](data) # test

                model_writer_list[m].add_scalar("Test/Reconstruction Error", recon_loss.item(), epoch)
                model_writer_list[m].add_scalar("Test/Regularizer", reg_loss.item(), epoch)
                model_writer_list[m].add_scalar("Test/Total Loss" , test_loss.item(), epoch)

        # After the completion of training
        if epoch % 5 == 0 or all(model_stop): 
            # Generation
            model_gen_list = [model.generate(gen_N).detach() for model in best_model_list]

            visualization = visualize_density(
                model_title_list, model_gen_list, xlim
            )

            generation_writer.add_figure(f"Generation_{exp_number}", visualization, epoch)
            visualization.savefig(f'{dirname}/generations/exp_{exp_number}_epoch{epoch}.png')

            mmd_result = [mmd_linear_bootstrap_test(gen[0:MMD_test_N], test_data[0:MMD_test_N], device = device, iteration = bootstrap_iter) for gen in model_gen_list]
            mmd_stat_list = [result[0] for result in mmd_result]
            mmd_p_value_list = [result[1] for result in mmd_result]

            for m in range(M) : 
                model_writer_list[m].add_scalar("Test/MMD score", mmd_stat_list[m], epoch)
                model_writer_list[m].add_scalar("Test/MMD p-value", mmd_p_value_list[m], epoch)

    np.savetxt(f'{dirname}/test_data.csv', test_data.cpu().numpy(), delimiter=',')
    [np.savetxt(f'{dirname}/{model_title_list[m]}.csv', model_gen_list[m].cpu().numpy(), delimiter = ',') for m in range(M)]

    return None