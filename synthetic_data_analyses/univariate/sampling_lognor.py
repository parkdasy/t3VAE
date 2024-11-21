import os
import torch
import random
import numpy as np

# from loss import log_t_normalizing_const
from util import make_reproducibility

# mu, cov

def lognor_sampling(N, mu, cov, nu, device) :
    mu = torch.tensor(mu, device=device) if not isinstance(mu, torch.Tensor) else mu.to(device)
    nu = torch.tensor(nu, device=device) if not isinstance(nu, torch.Tensor) else nu.to(device)
    lognor_dist = torch.distributions.log_normal.LogNormal(mu, nu)
    # par_dist = torch.distributions.pareto.Pareto(torch.tensor([mu]), torch.tensor([nu]))
    sample = lognor_dist.sample(sample_shape=torch.tensor([N]))
    
    return sample.to(device)

def sample_generation(device, SEED = None, 
                      K = 1, N = 1000, ratio_list = [1.0], mu_list = None, var_list = None, nu_list = None) : 
    if SEED is not None : 
        make_reproducibility(SEED)

    N_list = np.random.multinomial(N, ratio_list)
    result_list = [lognor_sampling(N_list[ind], mu_list[ind], var_list[ind], nu_list[ind], device) for ind in range(K)]
    result = torch.cat(result_list)
    shuffled_ind = torch.randperm(result.shape[0])
    return result[shuffled_ind]

import torch

def lognor_density(x, nu, mu=torch.ones(1), var=torch.ones(1, 1)):
    mu = torch.tensor(mu) if not isinstance(mu, torch.Tensor) else mu
    nu = torch.tensor(nu) if not isinstance(nu, torch.Tensor) else nu
    x = torch.tensor(x) if not isinstance(x, torch.Tensor) else x

    mask = x >= 0
    const_term = -torch.log(x)-torch.log(nu)-np.log(2 * np.pi)/2
    power_term = -(torch.log(x)-mu)**2/2/(nu**2)
    
    result = torch.where(mask, torch.exp(const_term + power_term), torch.tensor(0.0))
    return result


def lognor_density_contour(x, K, sample_nu_list, mu_list, var_list, ratio_list) : 
    output = 0
    for ind in range(K) : 
        output += ratio_list[ind] * lognor_density(x, sample_nu_list[ind], mu_list[ind], var_list[ind])
    return output
    
