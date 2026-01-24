import os
import math
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.reverse_diffusion import ReverseDiffusionUNet
from model.gnn import GraphNeuralNetwork
from model.utils import DependencyNormalization, FiLM


def map_pred_to_target_by_nearest_mz(
    pred_mz: torch.Tensor,
    pred_I: torch.Tensor,
    tgt_mz: torch.Tensor
):
    if pred_mz.dim() == 1:
        pred_mz = pred_mz.unsqueeze(0)
    if pred_I.dim() == 1:
        pred_I = pred_I.unsqueeze(0)
    if tgt_mz.dim() == 1:
        tgt_mz = tgt_mz.unsqueeze(0)

    B = max(pred_mz.size(0), tgt_mz.size(0))
    pred_mz = _expand_batch_to(pred_mz, B)
    pred_I = _expand_batch_to(pred_I, B)
    tgt_mz = _expand_batch_to(tgt_mz, B)

    dist = torch.cdist(pred_mz.unsqueeze(-1), tgt_mz.unsqueeze(-1), p=1)
    matched_idx = dist.argmin(dim=1)
    matched_I = torch.gather(pred_I, dim=1, index=matched_idx)
    return matched_I, matched_idx


def select_top_k(mz, intensity, k=5):
    if intensity.dim() == 1:
        intensity = intensity.unsqueeze(0)
    if mz.dim() == 1:
        mz = mz.unsqueeze(0)

    B, N = intensity.shape
    if mz.size(0) != B:
        if mz.size(0) == 1:
            mz = mz.expand(B, -1)
        else:
            raise ValueError(f"[select_top_k] Batch size mismatch: mz {mz.size()}, intensity {intensity.size()}")

    k = min(k, N)
    topk_intensity, indices = torch.topk(intensity, k=k, dim=-1)
    selected_mz = torch.gather(mz, 1, indices)
    return selected_mz, topk_intensity


def align_tensor(tensor, target_tensor, mode="nearest"):
    while tensor.dim() < target_tensor.dim():
        tensor = tensor.unsqueeze(0)

    if tensor.size(0) != target_tensor.size(0):
        tensor = tensor.expand(target_tensor.size(0), *tensor.shape[1:])

    if tensor.dim() == 2:
        if tensor.size(1) != target_tensor.size(1):
            tensor = F.interpolate(tensor.unsqueeze(1), size=target_tensor.size(1), mode=mode).squeeze(1)
    elif tensor.dim() == 3:
        if tensor.size(1) != target_tensor.size(1):
            tensor = F.interpolate(
                tensor.permute(0, 2, 1),
                size=target_tensor.size(1),
                mode=mode
            ).permute(0, 2, 1)
        if tensor.size(2) != target_tensor.size(2):
            tensor = F.interpolate(tensor, size=target_tensor.size(2), mode=mode)

    return tensor


def compute_complexity(molecular_sizes, adjacencies, dependency):
    batch_size = molecular_sizes.size(0)
    complexity = []
    for i in range(batch_size):
        n_atoms = molecular_sizes[i].item()
        adj = adjacencies[i]
        num_edges = adj.sum().item() / 2
        density = num_edges / (n_atoms + 1e-6)
        frag_factor = 1.0 - dependency[i].mean().item()
        score = (n_atoms / 50) * 0.4 + (density / 4) * 0.4 + frag_factor * 0.2
        complexity.append(min(score, 1.0))
    return torch.tensor(complexity, device=molecular_sizes.device)


def _expand_batch_to(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.size(0) == 1 and B > 1:
        t = t.expand(B, *t.shape[1:])
    return t


def _safe_nan_to_num(x: torch.Tensor, pos=1e6, neg=-1e6):
    return torch.nan_to_num(x, nan=0.0, posinf=pos, neginf=neg)

def _safe_std(x: torch.Tensor, eps: float = 1e-6):
    # unbiased=False가 핵심 (NaN std 방지)
    s = x.std(unbiased=False)
    s = _safe_nan_to_num(s)
    return s.clamp_min(eps)


def custom_noise_like(x, mode='gaussian', alpha=None, normalize=True):
    if alpha is None:
        alpha = 0.5

    if mode == 'gaussian':
        noise = torch.randn_like(x)
    elif mode == 'laplace':
        noise = torch.distributions.Laplace(0, 1).sample(x.shape).to(x.device)
    elif mode == 'exponential':
        noise = torch.distributions.Exponential(1.0).sample(x.shape).to(x.device) - 1.0
    elif mode == 'gamma':
        noise = torch.distributions.Gamma(2.0, 1.0).sample(x.shape).to(x.device) - 2.0
    elif mode == 'lognormal':
        noise = torch.distributions.LogNormal(0.0, 1.0).sample(x.shape).to(x.device) - math.exp(0.5)
    elif mode == 'hybrid_gamma':
        gamma = torch.distributions.Gamma(2.0, 1.0).sample(x.shape).to(x.device) - 2.0
        gaussian = torch.randn_like(x)
        noise = alpha * gamma + (1 - alpha) * gaussian
    elif mode == 'hybrid_laplace':
        laplace = torch.distributions.Laplace(0, 1).sample(x.shape).to(x.device)
        gaussian = torch.randn_like(x)
        noise = alpha * gaussian + (1 - alpha) * laplace
    elif mode == 'hybrid_exponential':
        exponential = torch.distributions.Exponential(1.0).sample(x.shape).to(x.device) - 1.0
        gaussian = torch.randn_like(x)
        noise = alpha * gaussian + (1 - alpha) * exponential
    else:
        raise ValueError(f"Unknown noise mode: {mode}")

    if normalize:
        noise = _safe_nan_to_num(noise)
        mean = noise.mean()
        std = _safe_std(noise, eps=1e-6)
        noise = (noise - mean) / std

    return noise


def entmax15(inputs: torch.Tensor, dim: int = -1, n_iter: int = 50):
    x = inputs
    if dim != -1:
        x = x.transpose(dim, -1)

    shape = x.shape
    x = x.reshape(-1, shape[-1])

    tau_lo = (x.min(dim=1, keepdim=True).values - x.abs().max(dim=1, keepdim=True).values - 1.0)
    tau_hi = x.max(dim=1, keepdim=True).values

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) * 0.5
        z = (x - tau_mid).clamp_min(0.0)
        f_mid = (z * z).sum(dim=1, keepdim=True) - 1.0
        go_right = (f_mid > 0)
        tau_lo = torch.where(go_right, tau_mid, tau_lo)
        tau_hi = torch.where(go_right, tau_hi, tau_mid)

    tau = (tau_lo + tau_hi) * 0.5
    p = (x - tau).clamp_min(0.0)
    p = p * p
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)

    p = p.reshape(shape)
    if dim != -1:
        p = p.transpose(dim, -1)
    return p


def sinkhorn_transport(cost, r, c, eps=0.05, iters=50, eps_stab=1e-8):
    B, Np, Nt = cost.shape
    K = torch.exp(-cost / (eps + 1e-12)).clamp_min(1e-12)

    u = torch.ones(B, Np, device=cost.device) / max(1, Np)
    v = torch.ones(B, Nt, device=cost.device) / max(1, Nt)

    for _ in range(iters):
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
        Kv = Kv.clamp_min(eps_stab)
        u = (r / Kv).clamp_max(1e6)

        Ku = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        Ku = Ku.clamp_min(eps_stab)
        v = (c / Ku).clamp_max(1e6)

        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    T = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return T


class DiffusionModel_Gaussian(nn.Module):
    def __init__(
        self,
        N_fingerprints,
        dim,
        num_layers,
        latent_dim,
        device,
        feature_dim=9,
        noise_mode='hybrid_gamma',
        loss_mode='balanced',
        mz_min=1.0,
        mz_max=426.0,
        multi_step_training=False,
        num_steps=3,
        normalize_pred_intensity=True,
        use_entmax=True
    ):
        super().__init__()
        self.device = device
        self.current_epoch = 0
        self.total_epochs = 100

        self.encoder = GraphNeuralNetwork(N_fingerprints, dim, num_layers, num_layers, device)
        self.feature_fc = nn.Linear(feature_dim, 50)
        self.linear_proj = nn.Linear(dim + 50, latent_dim)
        self.reverse_diffusion = ReverseDiffusionUNet(latent_dim=latent_dim, target_dim=latent_dim)

        self.dependency_film = FiLM(latent_dim, in_channels=latent_dim)
        self.intensity_film = FiLM(latent_dim, in_channels=1)

        self.dependency_norm = DependencyNormalization()
        self.latent_dim = latent_dim
        self.noise_mode = noise_mode
        self.loss_mode = loss_mode
        self.mz_min = mz_min
        self.mz_max = mz_max

        self.I_temp_start = 1.2
        self.I_temp_end = 0.7
        self.I_temp = self.I_temp_start
        self.use_entmax = use_entmax

        self.decoder_I_logits = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1)
        )
        self.decoder_I_scale = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1),
            nn.Softplus()
        )

        self.max_slots = 1024
        self.slot_embed = nn.Embedding(self.max_slots, self.latent_dim)

        self.decoder_mz = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1)
        )

        self.num_timesteps = 100
        self.betas = None
        self.alphas = None
        self.alpha_cumprod = None
        self.multi_step_training = multi_step_training
        self.num_steps = num_steps
        self.normalize_pred_intensity = normalize_pred_intensity

        self.mz_residual_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1)
        )
        self.ot_eps_init = 0.1
        self.ot_eps_min = 0.02

        self.ot_lambda_mz = 1.0
        self.ot_lambda_I = 1.0
        self.ot_eps = 0.05
        self.ot_iters = 60
        self.w_repulsion_soft = 2e-2
        self.w_repulsion_hard = 2e-2
        self.repulsion_margin = 0.05
        self.w_rank = 1e-2
        self.w_entropy = 5e-4
        self.w_T_entropy = 1e-3
        self.w_top1_unit = 1e-3
        self.w_cos = 1e-2
        self.lambda_coupling = 0.3
        self.coupling_weight_by = "target"
        self.coupling_p = 1
        self.ranking_margin = 0.03

    def set_epoch_info(self, current_epoch, total_epochs):
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        if total_epochs > 0:
            x = current_epoch / (total_epochs - 1)
            self.I_temp = self.I_temp_end + (self.I_temp_start - self.I_temp_end) * (math.cos(math.pi * x) ** 2)
            self.ot_eps = self.ot_eps_init * (1 - x) + self.ot_eps_min * x

    def _distribution_components(self, predicted_mz, predicted_intensity, target_mz, target_intensity):
        if predicted_mz.dim() == 1:
            predicted_mz = predicted_mz.unsqueeze(0)
        if predicted_intensity.dim() == 1:
            predicted_intensity = predicted_intensity.unsqueeze(0)
        if target_mz.dim() == 1:
            target_mz = target_mz.unsqueeze(0)
        if target_intensity.dim() == 1:
            target_intensity = target_intensity.unsqueeze(0)

        B = predicted_mz.size(0)
        target_mz = _expand_batch_to(target_mz, B)
        target_intensity = _expand_batch_to(target_intensity, B)
        predicted_intensity = _expand_batch_to(predicted_intensity, B)

        K = min(predicted_mz.size(1), target_mz.size(1))

        tgt_topk_mz, _ = select_top_k(target_mz, target_intensity, k=K)
        pred_topk_mz, _ = select_top_k(predicted_mz, target_intensity, k=K)
        tgt_topk_int, _ = select_top_k(target_intensity, target_intensity, k=K)
        pred_topk_int, _ = select_top_k(predicted_intensity, target_intensity, k=K)

        tgt_sorted_mz, tgt_idx = torch.sort(tgt_topk_mz, dim=-1)
        pred_sorted_mz = torch.gather(pred_topk_mz, 1, tgt_idx)

        p = torch.gather(F.softmax(tgt_topk_int, dim=-1), 1, tgt_idx)
        q = torch.gather(F.softmax(pred_topk_int, dim=-1), 1, tgt_idx)

        eps = 1e-8
        kl = torch.sum(p * torch.log((p + eps) / (q + eps)), dim=-1).mean()

        mz_range = self.mz_max - self.mz_min
        wass = ((p * torch.abs(pred_sorted_mz - tgt_sorted_mz)).sum(dim=-1) / (mz_range + 1e-8)).mean()
        return kl, wass

    def get_adaptive_beta_schedule(self, complexity_score, schedule_type='cosine'):
        steps = torch.linspace(0, 1, self.num_timesteps + 1, device=self.device)
        alphas = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, 0.0001, 0.999)
        scaling = 0.3 + complexity_score * 0.5
        betas = betas * scaling
        return betas

    def get_noise_alpha(self, power=2.0, min_alpha=0.01, fast_decay_point=0.9):
        if self.total_epochs == 0:
            return 1.0
        progress = self.current_epoch / self.total_epochs
        if progress >= fast_decay_point:
            alpha = min_alpha
        else:
            alpha = (1.0 - progress) ** power
            alpha = max(min_alpha, alpha)
        return alpha

    def q_sample(self, x_0, t, noise=None):
        def _nt(x):
            return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        x_0 = _nt(x_0)
        t = t.long()
        t = torch.clamp(t, 0, self.alpha_cumprod.numel() - 1)

        if noise is None:
            alpha = self.get_noise_alpha()
            noise = custom_noise_like(x_0, mode=self.noise_mode, alpha=alpha)

        noise = _nt(noise)

        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1)
        alpha_cumprod_t = _nt(alpha_cumprod_t).clamp(0.0, 1.0)

        out = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt((1.0 - alpha_cumprod_t).clamp_min(1e-12)) * noise
        out = _nt(out)
        return out, noise


    def p_sample(self, x_t, t, dependency):
        def _nt(x):
            return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        def safe_sqrt(x, eps=1e-12):
            return torch.sqrt(torch.clamp(x, min=eps))

        x_t = _nt(x_t)
        t = t.long()
        t = torch.clamp(t, 0, self.alpha_cumprod.numel() - 1)

        dependency = _nt(dependency).clamp(0.0, 1.0)
        dependency = align_tensor(dependency, x_t)
        dependency = _nt(dependency).clamp(0.0, 1.0)

        input_tensor = torch.cat([x_t, dependency], dim=2)
        predicted_noise = self.reverse_diffusion(input_tensor, t)
        predicted_noise = _nt(predicted_noise)

        beta_t = _nt(self.betas[t].view(-1, 1, 1)).clamp(0.0, 0.999)
        alpha_t = _nt(self.alphas[t].view(-1, 1, 1)).clamp(1e-12, 1.0)
        alpha_cumprod_t = _nt(self.alpha_cumprod[t].view(-1, 1, 1)).clamp(0.0, 1.0)

        denom = safe_sqrt(1.0 - alpha_cumprod_t)
        mu = (1.0 / safe_sqrt(alpha_t)) * (x_t - (1.0 - alpha_t) / denom * predicted_noise)
        mu = _nt(mu)

        if int(t[0].item()) > 0:
            eps_noise = torch.randn_like(x_t)
            eps_noise = _nt(eps_noise)
            sigma = safe_sqrt(beta_t)
            x_t_minus_1 = mu + sigma * eps_noise
        else:
            x_t_minus_1 = mu

        x_t_minus_1 = _nt(x_t_minus_1)
        return x_t_minus_1, predicted_noise

    def forward(self, fingerprints, adjacencies, molecular_sizes, dependency, feature_tensor, t):
        dependency = _safe_nan_to_num(dependency).clamp(0.0, 1.0)
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        if self.betas is None:
            complexity_score = compute_complexity(molecular_sizes, adjacencies, dependency)
            complexity_score = _safe_nan_to_num(complexity_score).clamp(0.0, 1.0)
            avg_complexity = float(complexity_score.mean().item())
            self.betas = self.get_adaptive_beta_schedule(avg_complexity)
            self.alphas = 1.0 - self.betas
            self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        mol_vec = self.encoder(fingerprints, adjacencies, molecular_sizes, dependency, feature_tensor)
        feature_emb = self.feature_fc(feature_tensor)
        feature_emb = align_tensor(feature_emb, mol_vec)
        mol_vec = torch.cat([mol_vec, feature_emb], dim=-1)
        z = self.linear_proj(mol_vec)

        x_t, noise = self.q_sample(z, t)
        predicted_mz_list = []
        predicted_intensity_list = []
        predicted_noise_list = []
        x_t_list = []
        x_t_minus_1_list = []

        for step in range(self.num_steps if self.multi_step_training else 1):
            current_t = torch.clamp(t - step, min=0)
            x_t_minus_1, predicted_noise = self.p_sample(x_t, current_t, dependency)

            x_dep = self.dependency_film(x_t_minus_1, dependency)

            with torch.no_grad():
                temp_logits = x_dep.mean(dim=-1)
                rank_idx = torch.argsort(-temp_logits, dim=1)
            inv_rank = torch.empty_like(rank_idx)
            inv_rank.scatter_(
                1,
                rank_idx,
                torch.arange(rank_idx.size(1), device=rank_idx.device).unsqueeze(0).expand_as(rank_idx)
            )
            slot_feat = self.slot_embed(inv_rank.clamp_max(self.max_slots - 1))

            I_logits = self.decoder_I_logits(torch.cat([x_dep, slot_feat], dim=-1)).squeeze(-1)
            I_logits = I_logits / (1.0 + I_logits.abs().mean(dim=1, keepdim=True))
            I_logits = I_logits.clamp(-50, 50)

            if self.use_entmax:
                I_logits_scaled = I_logits / max(1e-3, self.I_temp)
                p = entmax15(I_logits_scaled, dim=1)
            else:
                p = F.softmax(I_logits / max(1e-3, self.I_temp), dim=1)

            s = self.decoder_I_scale(x_dep.mean(dim=1))
            intensity_raw = s * p
            I_norm, _ = self.normalize_by_top1(intensity_raw)

            x_cond = self.intensity_film(x_dep, I_norm.unsqueeze(-1)) + slot_feat
            mz = self.decoder_mz(x_cond).squeeze(-1)
            mz = mz + self.mz_residual_head(x_cond).squeeze(-1)

            predicted_mz_list.append(mz)
            predicted_intensity_list.append(intensity_raw)
            predicted_noise_list.append(predicted_noise)
            x_t_list.append(x_t)
            x_t_minus_1_list.append(x_t_minus_1)

            x_t = x_t_minus_1

        return predicted_mz_list, predicted_intensity_list, noise, predicted_noise_list, x_t_list, x_t_minus_1_list, z

    def normalize_mz(self, mz, mz_min, mz_max):
        return (mz - mz_min) / (mz_max - mz_min + 1e-8)

    @staticmethod
    def normalize_by_top1(x, eps=1e-8):
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return x / (maxv + eps), maxv

    def intensity_upper_bound_penalty(self, predicted_intensity, max_val=1.2, weight=1.0):
        excess = F.relu(predicted_intensity - max_val)
        return weight * torch.mean(excess ** 2)

    def repulsion_loss(self, mz, tau=1.0):
        if mz.dim() == 1:
            mz = mz.unsqueeze(0)
        B, N = mz.shape
        diffs = mz.unsqueeze(2) - mz.unsqueeze(1)
        mask = 1 - torch.eye(N, device=mz.device).unsqueeze(0)
        rep = torch.exp(-torch.abs(diffs) / (tau + 1e-8)) * mask
        return rep.sum(dim=(1, 2)).mean() / max(1, N * (N - 1))

    def repulsion_margin_loss(self, mz, margin=0.05):
        if mz.dim() == 1:
            mz = mz.unsqueeze(0)
        mz_n = self.normalize_mz(mz, self.mz_min, self.mz_max)
        diff = torch.abs(mz_n.unsqueeze(2) - mz_n.unsqueeze(1))
        B, N = mz_n.shape
        mask = 1 - torch.eye(N, device=mz.device).unsqueeze(0)
        loss = F.relu(margin - diff) * mask
        return loss.sum() / (B * max(1, N * (N - 1)))

    def ranking_loss(self, pred_intensity, target_intensity, margin=0.03):
        if pred_intensity.dim() == 1:
            pred_intensity = pred_intensity.unsqueeze(0)
        if target_intensity.dim() == 1:
            target_intensity = target_intensity.unsqueeze(0)

        B, K = target_intensity.shape
        pi = pred_intensity.unsqueeze(2) - pred_intensity.unsqueeze(1)
        ti = target_intensity.unsqueeze(2) - target_intensity.unsqueeze(1)
        mask = (ti > 0).float()

        target_std = target_intensity.std(dim=1, keepdim=True)
        dynamic_margin = torch.clip(target_std * 0.05, 0.02, 0.08)
        dynamic_margin = dynamic_margin.unsqueeze(2).expand(B, K, K)

        loss = F.relu(dynamic_margin - pi) * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def mz_intensity_coupling_loss(
        self,
        pred_mz_topk: torch.Tensor,
        target_mz_topk: torch.Tensor,
        pred_int_topk: torch.Tensor,
        target_int_topk: torch.Tensor,
        weight_by: str = "target",
        p: int = 1
    ) -> torch.Tensor:
        if pred_mz_topk.dim() == 1:
            pred_mz_topk = pred_mz_topk.unsqueeze(0)
        if target_mz_topk.dim() == 1:
            target_mz_topk = target_mz_topk.unsqueeze(0)
        if pred_int_topk.dim() == 1:
            pred_int_topk = pred_int_topk.unsqueeze(0)
        if target_int_topk.dim() == 1:
            target_int_topk = target_int_topk.unsqueeze(0)

        if weight_by == "target":
            w = F.softmax(target_int_topk, dim=-1)
        elif weight_by == "pred":
            w = F.softmax(pred_int_topk, dim=-1)
        elif weight_by == "both":
            w = 0.5 * (F.softmax(target_int_topk, dim=-1) + F.softmax(pred_int_topk, dim=-1))
        else:
            raise ValueError(f"Unknown weight_by: {weight_by}")

        mz_diff = self.normalize_mz(pred_mz_topk, self.mz_min, self.mz_max) - \
                  self.normalize_mz(target_mz_topk, self.mz_min, self.mz_max)

        if p == 1:
            d = torch.abs(mz_diff)
        elif p == 2:
            d = mz_diff ** 2
        else:
            raise ValueError(f"Unsupported p (use 1 or 2): {p}")

        return (w * d).sum(dim=-1).mean()

    def ot_pair_loss(self, pred_mz, pred_I, tgt_mz, tgt_I, lam_mz=1.0, lam_I=1.0, eps=None, iters=None):
        eps = eps or self.ot_eps
        iters = iters or int(40 + 20 * (1 - self.current_epoch / max(1, self.total_epochs)))

        if pred_mz.dim() == 1:
            pred_mz = pred_mz.unsqueeze(0)
        if pred_I.dim() == 1:
            pred_I = pred_I.unsqueeze(0)
        if tgt_mz.dim() == 1:
            tgt_mz = tgt_mz.unsqueeze(0)
        if tgt_I.dim() == 1:
            tgt_I = tgt_I.unsqueeze(0)

        B = pred_mz.size(0)
        tgt_mz = _expand_batch_to(tgt_mz, B)
        tgt_I = _expand_batch_to(tgt_I, B)

        pred_I = torch.nan_to_num(pred_I, nan=0.0, posinf=0.0, neginf=0.0)
        tgt_I  = torch.nan_to_num(tgt_I,  nan=0.0, posinf=0.0, neginf=0.0)
        pred_I = _safe_nan_to_num(pred_I)
        tgt_I  = _safe_nan_to_num(tgt_I)

        pred_I = pred_I.clamp_min(0.0)
        tgt_I  = tgt_I.clamp_min(0.0)
    
        mz_range = (self.mz_max - self.mz_min) + 1e-8
        dmz = torch.cdist(pred_mz.unsqueeze(-1), tgt_mz.unsqueeze(-1), p=1) / mz_range
        dI = torch.cdist(pred_I.unsqueeze(-1), tgt_I.unsqueeze(-1), p=1)
        cost = lam_mz * dmz + lam_I * dI

        r = F.softmax(pred_I, dim=1)
        c = F.softmax(tgt_I, dim=1)

        T = sinkhorn_transport(cost, r, c, eps=eps, iters=iters)
        ot_loss = (T * cost).sum(dim=(1, 2)).mean()
        return ot_loss, T

    def loss_fn(
        self,
        predicted_mz_list,
        predicted_intensity_list,
        target_mz,
        target_intensity,
        z_0,
        x_t_list,
        x_t_minus_1_list,
        noise,
        predicted_noise_list,
        t,
        dependency,
        target_norm_intensity=None,
        k_supervise=None
    ):
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=1.0)

        total_loss = 0
        total_mz_loss = 0
        total_intensity_loss = 0
        total_intensity_norm_loss = 0
        total_top1_unit_loss = 0
        total_noise_pred_loss = 0
        total_recon_loss = 0
        total_l2_reg = 0
        total_ot_loss = 0
        total_top1_mz_loss = 0
        total_penalty_loss = 0
        total_kl_component = 0.0
        total_wass_component = 0.0
        total_coupling_loss = 0.0

        num_steps = len(predicted_mz_list)

        for step in range(num_steps):
            pred_mz = predicted_mz_list[step]
            pred_I = predicted_intensity_list[step]
            predicted_noise = predicted_noise_list[step]
            x_t = x_t_list[step]
            x_t_minus_1 = x_t_minus_1_list[step]

            if pred_mz.dim() == 1:
                pred_mz = pred_mz.unsqueeze(0)
            if target_mz.dim() == 1:
                target_mz = target_mz.unsqueeze(0)
            if target_intensity.dim() == 1:
                target_intensity = target_intensity.unsqueeze(0)

            B = pred_mz.size(0)
            tgt_mz = _expand_batch_to(target_mz, B)
            tgt_I = _expand_batch_to(target_intensity, B)

            matched_pred_I, _ = map_pred_to_target_by_nearest_mz(pred_mz, pred_I, tgt_mz)

            K_base = tgt_I.size(1)
            K = min(K_base, k_supervise) if k_supervise is not None else K_base
            _, topk_indices = torch.topk(tgt_I, k=K, dim=-1)

            tgt_I_topk = torch.gather(tgt_I, 1, topk_indices)
            matched_I_topk = torch.gather(matched_pred_I, 1, topk_indices)

            matched_pred_I, _ = map_pred_to_target_by_nearest_mz(pred_mz, pred_I, tgt_mz)
            eps = 1e-8
            p_prob = (matched_I_topk.clamp_min(0.0) + eps)
            p_prob = p_prob / (p_prob.sum(dim=1, keepdim=True) + eps)

            q_prob = (tgt_I_topk.clamp_min(0.0) + eps)
            q_prob = q_prob / (q_prob.sum(dim=1, keepdim=True) + eps)

            intensity_loss = torch.sum(q_prob * torch.log((q_prob + eps) / (p_prob + eps)), dim=1).mean()
            total_intensity_loss += intensity_loss

            ot_loss, T = self.ot_pair_loss(
                pred_mz=pred_mz, pred_I=pred_I,
                tgt_mz=tgt_mz, tgt_I=tgt_I,
                lam_mz=self.ot_lambda_mz, lam_I=self.ot_lambda_I,
                eps=self.ot_eps, iters=self.ot_iters
            )
            total_ot_loss += ot_loss

            def _entropy(p, eps=1e-8):
                p = p.clamp_min(eps)
                return -(p * p.log()).sum(dim=-1)

            row_denom = T.sum(dim=2, keepdim=True) + 1e-8
            row = T / row_denom

            col_denom = T.sum(dim=1) + 1e-8
            col = T.transpose(1, 2) / col_denom[:, :, None]

            T_entropy_bonus = _entropy(row).mean() + _entropy(col).mean()

            matched_pred_I, matched_idx = map_pred_to_target_by_nearest_mz(pred_mz, pred_I, tgt_mz)
            _, topk_indices = torch.topk(tgt_I, k=K, dim=-1)
            tgt_I_topk = torch.gather(tgt_I, 1, topk_indices)
            matched_I_topk = torch.gather(matched_pred_I, 1, topk_indices)
            matched_idx_topk = torch.gather(matched_idx, 1, topk_indices)
            pred_mz_topk = torch.gather(pred_mz, 1, matched_idx_topk)
            tgt_mz_topk = torch.gather(tgt_mz, 1, topk_indices)
            pred_int_topk = matched_I_topk
            tgt_int_topk = tgt_I_topk

            pred_mz_norm = self.normalize_mz(pred_mz_topk, self.mz_min, self.mz_max)
            target_mz_norm = self.normalize_mz(tgt_mz_topk, self.mz_min, self.mz_max)
            mz_loss = 1e2 * huber_loss(pred_mz_norm, target_mz_norm)
            total_mz_loss += mz_loss

            pred_int_topk_norm, _ = self.normalize_by_top1(pred_int_topk)
            if target_norm_intensity is not None:
                if target_norm_intensity.dim() == 1:
                    target_norm_intensity = target_norm_intensity.unsqueeze(0)
                if target_norm_intensity.size(0) != pred_mz.size(0):
                    if target_norm_intensity.size(0) == 1:
                        target_norm_intensity = target_norm_intensity.expand(pred_mz.size(0), -1)
                    else:
                        raise ValueError(
                            f"[loss_fn] Batch mismatch: target_norm_intensity {target_norm_intensity.size()} vs pred_mz {pred_mz.size()}"
                        )
                target_norm_topk = torch.gather(target_norm_intensity, 1, topk_indices)
            else:
                target_norm_topk, _ = self.normalize_by_top1(tgt_I_topk)

            intensity_norm_loss = 1e-2 * huber_loss(pred_int_topk_norm, target_norm_topk)
            total_intensity_norm_loss += intensity_norm_loss

            top1_unit_loss = mse_loss(
                pred_int_topk_norm.max(dim=1).values,
                torch.ones_like(pred_int_topk_norm.max(dim=1).values)
            )
            total_top1_unit_loss += top1_unit_loss

            coupling_loss = self.mz_intensity_coupling_loss(
                pred_mz_topk=pred_mz_topk,
                target_mz_topk=tgt_mz_topk,
                pred_int_topk=pred_int_topk,
                target_int_topk=tgt_int_topk,
                weight_by=self.coupling_weight_by,
                p=self.coupling_p
            )
            total_coupling_loss += coupling_loss

            ub_penalty = self.intensity_upper_bound_penalty(pred_I, max_val=1.2, weight=1.0)
            rep_soft = self.repulsion_loss(pred_mz, tau=1.0)
            rep_hard = self.repulsion_margin_loss(pred_mz, margin=self.repulsion_margin)
            rank = self.ranking_loss(pred_int_topk, tgt_int_topk, margin=self.ranking_margin)

            p_hat = (pred_I / (pred_I.sum(dim=1, keepdim=True) + 1e-8)).clamp_min(1e-8)
            ent_bonus = (-(p_hat * torch.log(p_hat)).sum(dim=1).mean())

            penalty_loss = (
                ub_penalty
                + self.w_rank * rank
                + self.w_repulsion_soft * rep_soft
                + self.w_repulsion_hard * rep_hard
                + self.w_entropy * ent_bonus
                - self.w_T_entropy * T_entropy_bonus
            )
            total_penalty_loss += penalty_loss

            top1_index = tgt_I.argmax(dim=1)
            top1_target_mz = tgt_mz.gather(1, top1_index.unsqueeze(1)).squeeze(1)
            top1_predicted_mz = pred_mz.gather(1, top1_index.unsqueeze(1)).squeeze(1)
            top1_target_mz_norm = self.normalize_mz(top1_target_mz, self.mz_min, self.mz_max)
            top1_predicted_mz_norm = self.normalize_mz(top1_predicted_mz, self.mz_min, self.mz_max)
            top1_mz_loss = huber_loss(top1_predicted_mz_norm, top1_target_mz_norm)
            total_top1_mz_loss += top1_mz_loss

            noise_pred_loss = 1e-1 * mse_loss(predicted_noise, noise)
            recon_loss = mse_loss(x_t_minus_1.mean(dim=-1), z_0.mean(dim=-1))
            l2_reg = 1e-5 * sum(torch.norm(p, 2) for p in self.parameters())
            total_noise_pred_loss += noise_pred_loss
            total_recon_loss += recon_loss
            total_l2_reg += l2_reg

            kl_comp, wass_comp = self._distribution_components(pred_mz, pred_I, tgt_mz, tgt_I)
            total_kl_component += kl_comp
            total_wass_component += wass_comp

            K = min(tgt_I.size(1), pred_I.size(1))
            _, topk_idx = torch.topk(tgt_I, K, dim=-1)
            tgt_I_topk = torch.gather(tgt_I, 1, topk_idx)
            matched_pred_I, _ = map_pred_to_target_by_nearest_mz(pred_mz, pred_I, tgt_mz)
            pred_I_topk = torch.gather(matched_pred_I, 1, topk_idx)

            cos = F.cosine_similarity(pred_I_topk, tgt_I_topk, dim=1, eps=1e-8)
            cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
            cos_corr_loss = 1 - cos.mean()

            total_loss += (
                ot_loss
                + intensity_loss
                + mz_loss + intensity_norm_loss
                + self.w_top1_unit * top1_unit_loss
                + noise_pred_loss + recon_loss + l2_reg
                + top1_mz_loss + penalty_loss
                + self.lambda_coupling * coupling_loss
                + self.w_cos * cos_corr_loss
            )

        num_steps = max(1, num_steps)
        return (
            total_loss / num_steps,
            total_mz_loss / num_steps,
            total_intensity_loss / num_steps,
            total_intensity_norm_loss / num_steps,
            total_top1_unit_loss / num_steps,
            total_noise_pred_loss / num_steps,
            total_recon_loss / num_steps,
            total_l2_reg / num_steps,
            total_ot_loss / num_steps,
            total_top1_mz_loss / num_steps,
            total_penalty_loss / num_steps,
            total_kl_component / num_steps,
            total_wass_component / num_steps,
            total_coupling_loss / num_steps
        )

    @torch.no_grad()
    def predict_topk_peaks(
        self,
        fingerprints,
        adjacencies,
        molecular_sizes,
        dependency,
        feature_tensor,
        steps=None,
        k=3,
        t=None,
        normalize_intensity=True,
        temperature: float = 1.2
    ):
        if steps is None:
            steps = self.num_timesteps - 1
        if t is None:
            t = torch.full((fingerprints.size(0),), steps, dtype=torch.long, device=self.device)

        out = self.forward(fingerprints, adjacencies, molecular_sizes, dependency, feature_tensor, t)
        predicted_mz_list, predicted_intensity_list = out[0], out[1]

        pred_mz = predicted_mz_list[-1]
        pred_intensity = predicted_intensity_list[-1]

        pred_intensity = pred_intensity / (pred_intensity.abs().mean(dim=1, keepdim=True) + 1e-8)
        pred_intensity = pred_intensity / temperature
        pred_intensity = pred_intensity.clamp_min(0.0)

        topk_mz, topk_int = select_top_k(pred_mz, pred_intensity, k=k)

        if normalize_intensity or self.normalize_pred_intensity:
            topk_int, _ = self.normalize_by_top1(topk_int)

        return topk_mz, topk_int
