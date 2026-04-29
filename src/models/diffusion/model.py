import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model for financial returns.
    Conditioned on historical context H.
    """
    def __init__(
        self,
        target_dim,
        context_dim,
        hidden_dim=128,
        predict_type='noise',
        context_seq_len: int | None = None,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.predict_type = predict_type  # 'noise' or 'x0'
        self.use_context_sequence = context_seq_len is not None
        self.context_seq_len = context_seq_len
        self.context_dim = context_dim

        # Time embedding for diffusion step t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.use_context_sequence:
            self.context_encoder = nn.LSTM(
                input_size=context_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.context_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        else:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Main denoising network
        # Inputs: noisy_target (target_dim) + time_emb (hidden_dim) + context_emb (hidden_dim)
        self.denoise_net = nn.Sequential(
            nn.Linear(target_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def _encode_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.use_context_sequence:
            if context.ndim == 4:
                batch, seq_len, _, _ = context.shape
                context = context.view(batch, seq_len, -1)
            elif context.ndim != 3:
                raise ValueError("Expected 3D or 4D context for sequential encoder.")
            context_out, _ = self.context_encoder(context)
            return self.context_proj(context_out[:, -1, :])

        if context.ndim != 2:
            context = context.view(context.shape[0], -1)
        return self.context_mlp(context)

    def forward(self, x_noisy, t, context):
        # x_noisy: [batch, target_dim]
        # t: [batch, 1]
        # context: [batch, context_dim] or [batch, L, n_features] or [batch, L, n_assets, n_channels]

        t_emb = self.time_mlp(t)
        c_emb = self._encode_context(context)

        combined = torch.cat([x_noisy, t_emb, c_emb], dim=1)
        return self.denoise_net(combined)
        
    def predict_x0(self, x_noisy, t, pred, alphas_cumprod):
        """
        Get predicted x0 from model output.
        pred is either noise or x0 depending on predict_type.
        """
        if self.predict_type == 'noise':
            alpha_cumprod = alphas_cumprod[t][:, None]
            return (x_noisy - torch.sqrt(1 - alpha_cumprod) * pred) / torch.sqrt(alpha_cumprod)
        elif self.predict_type == 'x0':
            return pred
        else:
            raise ValueError(f"Unknown predict_type: {self.predict_type}")

class DiffusionProcess:
    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02, schedule='linear', device='cpu'):
        self.num_steps = num_steps
        self.device = device
        self.schedule = schedule
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_steps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def _cosine_beta_schedule(self, num_steps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
        
    def add_noise(self, x_start, t):
        """Forward diffusion: q(x_t | x_0)"""
        noise = torch.randn_like(x_start)
        
        # alpha_t_bar = cumprod(alpha_1...t)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None]
        
        x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_noisy, noise, x_start

    @torch.no_grad()
    def sample(self, model, context, shape):
        """Reverse diffusion: p(x_0 | x_T, context)"""
        batch_size = shape[0]
        x = torch.randn(shape).to(self.device)
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size, 1), i, device=self.device, dtype=torch.float32)
            
            # Predict
            pred = model(x, t, context)
            pred_x0 = model.predict_x0(x, t, pred, self.alphas_cumprod)
            
            # Compute noise from pred_x0
            alpha_cumprod = self.alphas_cumprod[i]
            pred_noise = (x - torch.sqrt(alpha_cumprod) * pred_x0) / torch.sqrt(1 - alpha_cumprod)
            
            # Update x using standard DDPM with predicted noise
            alpha = self.alphas[i]
            beta = self.betas[i]
            
            sigma = torch.sqrt(beta) if i > 0 else 0
            
            mean = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            x = mean + sigma * torch.randn_like(x)
            
        return x
