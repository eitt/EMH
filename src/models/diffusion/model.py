import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model for financial returns.
    Conditioned on historical context H.
    """
    def __init__(self, target_dim, context_dim, hidden_dim=128):
        super().__init__()
        self.target_dim = target_dim
        
        # Time embedding for diffusion step t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context processing (could be LSTM/Transformer, using MLP for baseline reproduction/stability)
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main denoising network
        # Inputs: noisy_target (target_dim) + time_emb (hidden_dim) + context_emb (hidden_dim)
        self.denoise_net = nn.Sequential(
            nn.Linear(target_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim)
        )
        
    def forward(self, x_noisy, t, context):
        # x_noisy: [batch, target_dim]
        # t: [batch, 1]
        # context: [batch, context_dim]
        
        t_emb = self.time_mlp(t)
        c_emb = self.context_mlp(context)
        
        combined = torch.cat([x_noisy, t_emb, c_emb], dim=1)
        return self.denoise_net(combined)

class DiffusionProcess:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def add_noise(self, x_start, t):
        """Forward diffusion: q(x_t | x_0)"""
        noise = torch.randn_like(x_start)
        
        # alpha_t_bar = cumprod(alpha_1...t)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None]
        
        x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample(self, model, context, shape):
        """Reverse diffusion: p(x_0 | x_T, context)"""
        batch_size = shape[0]
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size, 1), i, device=self.device, dtype=torch.float32)
            
            # Predict noise
            predicted_noise = model(x, t, context)
            
            # Update x
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            sigma = torch.sqrt(beta) if i > 0 else 0
            
            # Mean update logic from DDPM
            mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            x = mean + sigma * torch.randn_like(x)
            
        return x
