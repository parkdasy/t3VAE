import os
import torch
import random
import numpy as np

# from loss import log_t_normalizing_const
from util import make_reproducibility

# mu - scale(x_m), nu - shape(alpha)

def par_sampling(N, mu, cov, nu, device) :
    mu = torch.tensor(mu, device=device) if not isinstance(mu, torch.Tensor) else mu.to(device)
    nu = torch.tensor(nu, device=device) if not isinstance(nu, torch.Tensor) else nu.to(device)
    par_dist = torch.distributions.pareto.Pareto(mu, nu)
    # par_dist = torch.distributions.pareto.Pareto(torch.tensor([mu]), torch.tensor([nu]))
    sample = par_dist.sample(sample_shape=torch.tensor([N]))
    
    return sample.to(device)

def sample_generation(device, SEED = None, 
                      K = 1, N = 1000, ratio_list = [1.0], mu_list = None, var_list = None, nu_list = None) : 
    if SEED is not None : 
        make_reproducibility(SEED)

    N_list = np.random.multinomial(N, ratio_list)
    result_list = [par_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    result = torch.cat(result_list)
    shuffled_ind = torch.randperm(result.shape[0])
    return result[shuffled_ind]

import torch

def par_density(x, nu, mu=torch.ones(1), var=torch.ones(1, 1)):
    nu = torch.tensor(nu) if not isinstance(nu, torch.Tensor) else nu
    mu = torch.tensor(mu) if not isinstance(mu, torch.Tensor) else mu
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x

    mask = x >= mu
    const_term = torch.log(nu) + nu * torch.log(mu)
    power_term = -(nu + 1) * torch.log(x)
    
    result = torch.where(mask, torch.exp(const_term + power_term), torch.tensor(0.0))
    return result


def par_density_contour(x, K, sample_nu_list, mu_list, var_list, ratio_list) : 
    output = 0
    for ind in range(K) : 
        output += ratio_list[ind] * par_density(x, sample_nu_list[ind], mu_list[ind], var_list[ind])
    return output
    
