"""Diffusion models: noise schedules, denoisers, and CFG sampling."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    if a.device != t.device:
        a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Discretize a beta schedule from a cumulative alpha function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class diffusion():
    """DDPM diffusion process with CFG-guided iterative denoising."""

    def __init__(self, timesteps, beta_start, beta_end, beta_sche, args):
        self.timesteps = timesteps
        self.args = args

        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=timesteps)
        elif beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=timesteps)
        elif beta_sche == 'sqrt':
            self.betas = torch.tensor(
                betas_for_alpha_bar(timesteps, lambda t: 1 - np.sqrt(t + 0.0001)),
            ).float()

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape,
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x_start, x_t, t, t_index):
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x_start.shape)
        noise = torch.randn_like(x_start)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, x, h, n_sampling_step, is_ddim=False):
        if x is None:
            x = torch.randn_like(h)
        device = h.device
        for t_index in reversed(range(0, n_sampling_step)):
            x_t = x
            t = torch.full((h.shape[0],), t_index, device=device, dtype=torch.long)
            x_start = (
                (1 - self.args.s_strength) * model_forward_uncon(x_t, t)
                + self.args.s_strength * model_forward(x_t, h, t)
            )
            x = self.p_sample(x_start, x_t, t, t_index)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Tenc(nn.Module):
    """Conditional denoiser fusing entity state, CFG condition, and timestep embedding."""

    def __init__(self, hidden_size, dropout, diffuser_type, device):
        super(Tenc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device

        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
            )

    def forward(self, x, h, step):
        t = self.step_mlp(step)
        return self.diffuser(torch.cat((x, h, t), dim=1))

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device)).expand_as(x)
        t = self.step_mlp(step)
        return self.diffuser(torch.cat((x, h, t), dim=1))

    def cacu_h(self, con, p):
        """Classifier-free guidance: replace the condition with the null embedding with probability p."""
        B, D = con.shape[0], con.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        mask = mask1d.view(B, 1).expand(B, D).to(self.device)
        null_emb = self.none_embedding(torch.tensor([0]).to(self.device))
        return con * mask + null_emb * (1 - mask)

    def predict(self, x, h, diff, n_sampling_step, is_ddim=False):
        return diff.sample(self.forward, self.forward_uncon, x, h, n_sampling_step, is_ddim)
