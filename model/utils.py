import torch.nn as nn
import torch.nn.functional as F


class DependencyNormalization(nn.Module):
    def __init__(self, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, intensity, dependency):
        if dependency.dim() == 1:
            dependency = dependency.unsqueeze(0)

        if dependency.dim() == 3:
            dependency = dependency.squeeze(-1)

        if dependency.size(1) != intensity.size(1):
            dependency = F.interpolate(
                dependency.unsqueeze(1),
                size=intensity.size(1),
                mode="nearest",
            ).squeeze(1)

        return (intensity / self.scale_factor) * dependency


class FiLM(nn.Module):
    def __init__(self, latent_dim, in_channels):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(latent_dim, in_channels),
            nn.SiLU(),
        )
        self.bias = nn.Sequential(
            nn.Linear(latent_dim, in_channels),
            nn.SiLU(),
        )

    def forward(self, x, dependency):
        dependency = dependency.clone()

        if dependency.dim() == 3:
            dependency = dependency.mean(dim=1)
        elif dependency.dim() == 2:
            dependency = dependency.squeeze(1)
            dependency = dependency.unsqueeze(0)
            dependency = dependency.expand(x.shape[0], -1)

        expected_dim = self.scale[0].in_features

        if dependency.shape[-1] != expected_dim:
            dependency = dependency[..., :expected_dim]
            dependency = F.pad(dependency, (0, expected_dim - dependency.shape[-1]), "constant", 0)

        scale = self.scale(dependency)
        bias = self.bias(dependency)

        if scale.dim() == 2:
            scale = scale.unsqueeze(1).expand(-1, x.shape[1], -1)
        if bias.dim() == 2:
            bias = bias.unsqueeze(1).expand(-1, x.shape[1], -1)

        return x * scale + bias
