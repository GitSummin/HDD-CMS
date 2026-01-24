import torch
import torch.nn as nn
from model.utils import FiLM


class ReverseDiffusionUNet(nn.Module):
    def __init__(self, latent_dim, target_dim, time_embed_dim=128, nhead=4, num_layers=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim + time_embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True, 
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self._dep_in_dim = None
        self.dependency_proj = None
        self.film = FiLM(latent_dim=latent_dim, in_channels=latent_dim + time_embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim),
        )

        self.output_proj = nn.Linear(256, target_dim)
        self.intermediate_features = None

    def _ensure_dep_proj(self, dep: torch.Tensor, latent_dim: int):
        in_dim = dep.size(-1)
        if (self.dependency_proj is None) or (self._dep_in_dim != in_dim):
            self.dependency_proj = nn.Linear(in_dim, latent_dim).to(dep.device)
            self._dep_in_dim = in_dim

    def forward(self, x, dependency, t=None):
        """
        x: (B, N, latent_dim)
        dependency: (B, N, *), (B, *), or (B, 1, *)
        t: (B,) or None
        """
        B, N, _ = x.shape

        if t is not None:
            t = t.to(x.device)
            t_embed = self.time_mlp(t.float().unsqueeze(-1))
            t_embed = t_embed.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([x, t_embed], dim=-1)

        x = x.transpose(0, 1)
        self.intermediate_features = x.detach().cpu()

        if dependency is None:
            dep = torch.zeros(B, N, 1, device=x.device)
        else:
            dep = dependency
            while dep.dim() < 3:
                dep = dep.unsqueeze(1)
            if dep.size(1) != N:
                dep = dep.expand(B, N, dep.size(-1))

        self._ensure_dep_proj(dep, latent_dim=x.size(-1) - 128)
        dep_proj = self.dependency_proj(dep.float())

        x = self.film(x, dep_proj)
        out = self.decoder(x)
        return out
