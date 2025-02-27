import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class TVAE(nn.Module) : 
    def __init__(self, n_dim=1, m_dim=1, nu=3, recon_sigma=1, reg_weight=1, num_layers=64, device='cpu'):
        super(TVAE, self).__init__()
        self.model_name = "t-VAE"

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.reg_weight = reg_weight
        self.num_layers = num_layers
        self.device = device

        # define encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.latent_mu = nn.Linear(num_layers, m_dim)
        self.latent_logvar = nn.Linear(num_layers, m_dim)

        # define decoder
        self.decoder = nn.Sequential(
            nn.Linear(m_dim, num_layers), 
            nn.LeakyReLU(), 
            nn.Linear(num_layers, num_layers), 
            nn.LeakyReLU()
        )
        self.decoder_mu = nn.Linear(num_layers, n_dim)
        self.decoder_loglambda = nn.Linear(num_layers, n_dim)
        self.decoder_lognu = nn.Linear(num_layers, n_dim)

    def encoder_reparameterize(self, mu, logvar) : 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def encode(self, x) : 
        x = self.encoder(x)
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        z = self.encoder_reparameterize(mu, logvar)

        return z, mu, logvar
    
    def decode(self, z) : 
        z = self.decoder(z)
        mu_theta = self.decoder_mu(z)
        loglambda = self.decoder_loglambda(z)
        lognu = self.decoder_lognu(z)
        return mu_theta, loglambda, lognu
    
    
    def recon_loss(self, x, mu_theta, loglambda, lognu) : 
        # Since VAE and t3VAE double each term in ELBO and gamma-loss respectively, we also doubled the reconstruction loss term. 
        # nll = torch.lgamma((torch.exp(lognu) + 1) / 2) - torch.lgamma(torch.exp(lognu) / 2)
        # nll -= (np.log(np.pi) + lognu) / 2
        # nll += loglambda / 2
        # nll -= (torch.exp(lognu) + 1) * torch.log(1 + (torch.exp(loglambda) * (x - mu_theta).pow(2) / torch.exp(lognu))) / 2
        
        # univariate loss
        # nll = 2 * (torch.lgamma((torch.exp(lognu) + 1) / 2) - torch.lgamma(torch.exp(lognu) / 2))
        # nll -= np.log(np.pi) + lognu
        # nll += loglambda
        # nll -= (torch.exp(lognu) + 1) * torch.log(1 + (torch.exp(loglambda) * (x - mu_theta).pow(2) / torch.exp(lognu)))
        
        # multivariate loss
        p = mu_theta.size(-1)
        nll = 2 * (torch.lgamma((torch.exp(lognu) + p) / 2) - torch.lgamma(torch.exp(lognu) / 2))
        nll -= p * (np.log(np.pi) + lognu)
        nll += p * loglambda
        nll -= (torch.exp(lognu) + p) * torch.log(1 + (torch.exp(loglambda) * (x - mu_theta).pow(2) / torch.exp(lognu)))
        
        return torch.mean(torch.sum(-1 * nll, dim=1))

    def reg_loss(self, mu, logvar) : 
        return torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1))
    
    def total_loss(self, x, mu_theta, loglambda, lognu, mu, logvar) : 
        recon = self.recon_loss(x, mu_theta, loglambda, lognu)
        reg = self.reg_loss(mu, logvar)

        return recon, reg, recon + self.reg_weight * reg
    

    def decoder_sampling(self, z) : 
        mu_theta, loglambda, lognu = self.decode(z)
        eps = torch.randn_like(mu_theta)

        nu = torch.exp(lognu)
        chi_dist = torch.distributions.chi2.Chi2(nu)
        v = chi_dist.sample(sample_shape=torch.tensor([1])).squeeze(0).to(self.device)

        std = torch.exp(-0.5 * loglambda)
        return mu_theta + std * eps * torch.sqrt(nu / v)

    def generate(self, N = 1000) : 
        prior = torch.randn(N, self.m_dim).to(self.device)

        return self.decoder_sampling(prior)
    
    def reconstruct(self, x) : 
        return self.decoder_sampling(self.encode(x)[0])

    def forward(self, x) : 
        enc_z, mu, logvar = self.encode(x)
        mu_theta, loglambda, lognu = self.decode(enc_z)
        # if x.shape[0] > 5000 : 
        #     print(f'TVAE nu : {torch.exp(lognu).flatten().detach().cpu().numpy()}')
        return self.total_loss(x, mu_theta, loglambda, lognu, mu, logvar)

        


