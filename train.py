import os
import sys
import time
import math
import warnings
import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from data import preprocess as pp
from model.diffusion import DiffusionModel_Gaussian

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Using a target size .*broadcasting.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.loss"
)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir="output/checkpoint",
        loss_file="output/loss_results.txt",
        model_args=None,
        k_supervise=5,
        start_epoch=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

        self.checkpoint_dir = checkpoint_dir
        self.loss_file = loss_file
        self.model_args = model_args or {}
        self.k_supervise = k_supervise
        self.start_epoch = int(start_epoch)

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.experiment_histories = {
            "only_kl": [],
            "only_wass": [],
            "kl_to_wass": [],
            "wass_to_kl": [],
            "raw_kl": [],
            "raw_wass": [],
            "weighted_kl": [],
            "weighted_wass": [],
        }
        self.epoch_times = []

    @staticmethod
    def _sigmoid_anneal(t: int, total_t: int, a: float = 10.0, b: float = 0.5) -> float:
        if total_t is None or total_t <= 0:
            return 0.5
        x = (t / total_t)
        return float(1.0 / (1.0 + math.exp(-a * (x - b))))

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def plot_experiment_losses(self):
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })

        def _get_series(name, fallback_name, L):
            s = self.experiment_histories.get(name, [])
            if not s or len(s) < L:
                s_fb = self.experiment_histories.get(fallback_name, [])
                if not s_fb or len(s_fb) < L:
                    return [0.0] * L
                return s_fb[:L]
            return s[:L]

        L = len(self.experiment_histories.get("only_kl", []))
        if L == 0:
            return

        epochs = list(range(1, L + 1))
        raw_kl = _get_series("raw_kl", "only_kl", L)
        raw_wass = _get_series("raw_wass", "only_wass", L)

        weighted_kl_hist = self.experiment_histories.get("weighted_kl", [])
        weighted_wass_hist = self.experiment_histories.get("weighted_wass", [])
        if len(weighted_kl_hist) >= L and len(weighted_wass_hist) >= L:
            w_kl = weighted_kl_hist[:L]
            w_wass = weighted_wass_hist[:L]
        else:
            a = 10.0
            b = 0.5
            total_epochs = max(getattr(self.model, "total_epochs", 0) or 0, L)
            sig_list = [
                self._sigmoid_anneal(t=e - 1, total_t=total_epochs, a=a, b=b)
                for e in epochs
            ]
            mode = getattr(self.model, "loss_mode", "balanced")
            if mode == "wass_to_kl":
                w_kl = [s * k for s, k in zip(sig_list, raw_kl)]
                w_wass = [(1 - s) * w for s, w in zip(sig_list, raw_wass)]
            elif mode == "kl_to_wass":
                w_kl = [(1 - s) * k for s, k in zip(sig_list, raw_kl)]
                w_wass = [s * w for s, w in zip(sig_list, raw_wass)]
            elif mode == "only_kl":
                w_kl, w_wass = raw_kl, [0.0] * L
            elif mode == "only_wass":
                w_kl, w_wass = [0.0] * L, raw_wass
            else:
                w_kl = [0.5 * k for k in raw_kl]
                w_wass = [0.5 * w for w in raw_wass]

        a = 10.0
        b = 0.5
        total_epochs = max(getattr(self.model, "total_epochs", 0) or 0, L)
        sig_list = [
            self._sigmoid_anneal(t=e - 1, total_t=total_epochs, a=a, b=b)
            for e in epochs
        ]

        fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
        ax1.plot(epochs, raw_kl, label="KL (raw avg)", linewidth=2)
        ax1.plot(epochs, raw_wass, label="Wass (raw avg)", linewidth=2)
        ax1.plot(epochs, w_kl, label="KL (weighted)", linewidth=2.5, linestyle="--")
        ax1.plot(epochs, w_wass, label="Wass (weighted)", linewidth=2.5, linestyle="--")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss / Weighted contribution")
        ax1.set_title(f"KL vs Wass (raw & weighted) — mode={self.model.loss_mode}")

        all_vals = raw_kl + raw_wass + w_kl + w_wass
        if all_vals and min(all_vals) >= 0:
            ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.plot(epochs, sig_list, label="sigma (anneal weight)", linewidth=1.8, alpha=0.7)
        ax2.set_ylabel("sigma(t)")
        ax2.set_ylim(-0.05, 1.05)

        mid_epoch = int(b * total_epochs) if total_epochs else (L // 2)
        if 1 <= mid_epoch <= L:
            ax1.axvline(mid_epoch, linestyle=":", linewidth=1.4)
            ymax = ax1.get_ylim()[1]
            ax1.text(mid_epoch + max(1, L // 100), ymax * 0.9, "σ midpoint", fontsize=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)

        fig.tight_layout()
        out_path = os.path.join(self.checkpoint_dir, f"kl_wass_loss_{self.model.loss_mode}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved KL/Wass loss plot to: {out_path}")

    def _compute_target_norm_by_top1(self, intensity: torch.Tensor) -> torch.Tensor:
        if intensity.dim() == 1:
            maxv = torch.max(intensity).clamp_min(1e-8)
            return intensity / maxv
        if intensity.dim() == 2:
            maxv, _ = torch.max(intensity, dim=1, keepdim=True)
            return intensity / (maxv + 1e-8)
        maxv, _ = torch.max(intensity, dim=-1, keepdim=True)
        return intensity / (maxv + 1e-8)

    def train(self, dataset_train, dataset_val, epochs: int):
        self.model.train()
        train_loss_history = []
        self.accumulated_features = []

        overall_t0 = time.perf_counter()

        max_epochs = int(epochs)
        start_epoch = max(0, int(self.start_epoch))
        if start_epoch >= max_epochs:
            print(f"[WARN] start_epoch({start_epoch}) >= max_epochs({max_epochs}). Nothing to do.")
            return

        for epoch in range(start_epoch, max_epochs):
            self.model.set_epoch_info(epoch, max_epochs)

            self._sync()
            epoch_t0 = time.perf_counter()

            total_train_loss = 0.0
            kl_epoch_sum = 0.0
            wass_epoch_sum = 0.0
            batch_count = 0

            sum_mz = sum_int = sum_intn = sum_top1u = 0.0
            sum_noise = sum_recon = sum_l2 = sum_dist = 0.0
            sum_top1mz = sum_pen = 0.0

            mz_sum = mz_sumsq = 0.0
            I_sum = I_sumsq = 0.0
            n_elems_mz = 0
            n_elems_I = 0

            for batch_idx, batch in enumerate(dataset_train):
                smiles, fingerprints, adjacencies, molecular_sizes, mz, intensity, dependency, feature_tensor = batch

                batch_size = fingerprints.size(0)
                t_batch = torch.randint(0, self.model.num_timesteps, (batch_size,), device=fingerprints.device)

                predicted_mz, predicted_intensity, noise, predicted_noise, x_t, x_t_minus_1, z = self.model(
                    fingerprints, adjacencies, molecular_sizes, dependency, feature_tensor, t_batch
                )

                _pm = predicted_mz[-1].detach() if isinstance(predicted_mz, list) else predicted_mz.detach()
                _pi = predicted_intensity[-1].detach() if isinstance(predicted_intensity, list) else predicted_intensity.detach()

                mz_sum += _pm.sum().item()
                mz_sumsq += (_pm ** 2).sum().item()
                I_sum += _pi.sum().item()
                I_sumsq += (_pi ** 2).sum().item()
                n_elems_mz += _pm.numel()
                n_elems_I += _pi.numel()

                target_norm_intensity = self._compute_target_norm_by_top1(intensity)

                (
                    loss,
                    mz_loss,
                    intensity_loss,
                    intensity_norm_loss,
                    top1_unit_loss,
                    noise_pred_loss,
                    recon_loss,
                    l2_reg_loss,
                    distribution_loss,
                    top1_mz_loss,
                    penalty_loss,
                    kl_comp,
                    wass_comp,
                    coupling_loss,
                ) = self.model.loss_fn(
                    predicted_mz_list=predicted_mz,
                    predicted_intensity_list=predicted_intensity,
                    target_mz=mz,
                    target_intensity=intensity,
                    z_0=z,
                    x_t_list=x_t,
                    x_t_minus_1_list=x_t_minus_1,
                    noise=noise,
                    predicted_noise_list=predicted_noise,
                    t=t_batch,
                    dependency=dependency,
                    target_norm_intensity=target_norm_intensity,
                    k_supervise=self.k_supervise,
                )

                sum_mz += mz_loss.item()
                sum_int += intensity_loss.item()
                sum_intn += intensity_norm_loss.item()
                sum_top1u += top1_unit_loss.item()
                sum_noise += noise_pred_loss.item()
                sum_recon += recon_loss.item()
                sum_l2 += l2_reg_loss.item()
                sum_dist += distribution_loss.item()
                sum_top1mz += top1_mz_loss.item()
                sum_pen += penalty_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step(epoch + batch_idx / max(1, len(dataset_train)))

                total_train_loss += loss.item()
                kl_epoch_sum += kl_comp.item()
                wass_epoch_sum += wass_comp.item()
                batch_count += 1

            train_loss_history.append(total_train_loss)

            print(f"[Epoch {epoch + 1}/{max_epochs}] Train Loss: {total_train_loss:.4f}")

            den = max(1, batch_count)
            avg_mz = sum_mz / den
            avg_int = sum_int / den
            avg_intn = sum_intn / den
            avg_top1u = sum_top1u / den
            avg_noise = sum_noise / den
            avg_recon = sum_recon / den
            avg_l2 = sum_l2 / den
            avg_dist = sum_dist / den
            avg_top1mz = sum_top1mz / den
            avg_pen = sum_pen / den

            print(
                "  ├─ Components (Train avg / epoch)\n"
                f"  │   MZ: {avg_mz:.4f} | INT: {avg_int:.4f} | INT_norm: {avg_intn:.4f} | Top1=1: {avg_top1u:.4f}\n"
                f"  │   Noise: {avg_noise:.4f} | Recon: {avg_recon:.4f} | L2: {avg_l2:.6f}\n"
                f"  │   Dist(MZ): {avg_dist:.4f} | Top1_mz: {avg_top1mz:.4f} | Penalty: {avg_pen:.4f}"
            )

            with open(self.loss_file, "a", encoding="utf-8") as fh:
                fh.write(
                    f"Epoch {epoch + 1}\t"
                    f"TrainLoss={total_train_loss:.6f}\t"
                    f"MZ={avg_mz:.6f}\tINT={avg_int:.6f}\tINTn={avg_intn:.6f}\tTop1=1={avg_top1u:.6f}\t"
                    f"Noise={avg_noise:.6f}\tRecon={avg_recon:.6f}\tL2={avg_l2:.8f}\t"
                    f"Dist={avg_dist:.6f}\tTop1_mz={avg_top1mz:.6f}\tPenalty={avg_pen:.6f}\n"
                    f"Coupling={coupling_loss:.6f}\n"
                )

            kl_avg = kl_epoch_sum / den
            wass_avg = wass_epoch_sum / den

            _sig = getattr(self.model, "sigmoid_annealing_weight", None)
            if callable(_sig):
                sig = _sig(self.model.current_epoch, self.model.total_epochs, a=10, b=0.5)
            else:
                sig = self._sigmoid_anneal(self.model.current_epoch, self.model.total_epochs, a=10, b=0.5)

            self.experiment_histories.setdefault("raw_kl", []).append(kl_avg)
            self.experiment_histories.setdefault("raw_wass", []).append(wass_avg)

            mode = getattr(self.model, "loss_mode", "balanced")
            if mode == "wass_to_kl":
                w_kl, w_wass = sig * kl_avg, (1.0 - sig) * wass_avg
            elif mode == "kl_to_wass":
                w_kl, w_wass = (1.0 - sig) * kl_avg, sig * wass_avg
            elif mode == "only_kl":
                w_kl, w_wass = kl_avg, 0.0
            elif mode == "only_wass":
                w_kl, w_wass = 0.0, wass_avg
            else:
                w_kl, w_wass = 0.5 * kl_avg, 0.5 * wass_avg

            self.experiment_histories.setdefault("weighted_kl", []).append(w_kl)
            self.experiment_histories.setdefault("weighted_wass", []).append(w_wass)

            self.experiment_histories["only_kl"].append(kl_avg)
            self.experiment_histories["only_wass"].append(wass_avg)
            self.experiment_histories["kl_to_wass"].append((1.0 - sig) * kl_avg + sig * wass_avg)
            self.experiment_histories["wass_to_kl"].append(sig * kl_avg + (1.0 - sig) * wass_avg)

            if n_elems_mz > 0:
                mz_mean = mz_sum / n_elems_mz
                mz_var = max(0.0, mz_sumsq / n_elems_mz - mz_mean ** 2)
                mz_std = mz_var ** 0.5
            else:
                mz_mean = mz_std = float("nan")

            if n_elems_I > 0:
                I_mean = I_sum / n_elems_I
                I_var = max(0.0, I_sumsq / n_elems_I - I_mean ** 2)
                I_std = I_var ** 0.5
            else:
                I_mean = I_std = float("nan")

            print(
                f"  ├─ Pred Stats (per-epoch): mz mean={mz_mean:.3f}, std={mz_std:.3f} | "
                f"I mean={I_mean:.4f}, std={I_std:.4f}"
            )

            with open(self.loss_file, "a", encoding="utf-8") as fh:
                fh.write(
                    f"PredStats: mz_mean={mz_mean:.6f}\tmz_std={mz_std:.6f}\t"
                    f"I_mean={I_mean:.6f}\tI_std={I_std:.6f}\n"
                )

            self.model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for batch in dataset_val:
                    smiles, fingerprints, adjacencies, molecular_sizes, mz, intensity, dependency, feature_tensor = batch
                    batch_size = fingerprints.size(0)
                    t_batch = torch.zeros(batch_size, dtype=torch.long, device=fingerprints.device)

                    predicted_mz, predicted_intensity, noise, predicted_noise, x_t, x_t_minus_1, z = self.model(
                        fingerprints, adjacencies, molecular_sizes, dependency, feature_tensor, t_batch
                    )

                    target_norm_intensity = self._compute_target_norm_by_top1(intensity)

                    (val_loss, *_rest) = self.model.loss_fn(
                        predicted_mz_list=predicted_mz,
                        predicted_intensity_list=predicted_intensity,
                        target_mz=mz,
                        target_intensity=intensity,
                        z_0=z,
                        x_t_list=x_t,
                        x_t_minus_1_list=x_t_minus_1,
                        noise=noise,
                        predicted_noise_list=predicted_noise,
                        t=t_batch,
                        dependency=dependency,
                        target_norm_intensity=target_norm_intensity,
                        k_supervise=self.k_supervise,
                    )
                    total_val_loss += val_loss.item()

            print(f"\n[Epoch {epoch + 1}/{max_epochs}] Validation Loss: {total_val_loss:.4f}\n")
            print(f"  └─ Diagnostics: KL(avg)={kl_avg:.5f} | Wass(avg)={wass_avg:.5f} | sig={sig:.4f}")

            self.model.train()

            if hasattr(self.model, "reverse_diffusion") and hasattr(self.model.reverse_diffusion, "intermediate_features"):
                features = self.model.reverse_diffusion.intermediate_features
                if features is not None and features.numel() > 0:
                    B, N, C = features.shape
                    self.accumulated_features.append(features.reshape(B, -1))

            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pth")
            torch.save({
                "model_args": self.model_args,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": int(epoch + 1),
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint to: {checkpoint_path}")

            self.plot_loss(train_loss_history)
            self.plot_experiment_losses()

            self._sync()
            epoch_sec = time.perf_counter() - epoch_t0
            self.epoch_times.append(epoch_sec)
            print(f"[Timing] Epoch {epoch + 1} time: {epoch_sec:.3f} sec")

        self._sync()
        total_training_sec = time.perf_counter() - overall_t0
        avg_epoch_sec = sum(self.epoch_times) / max(1, len(self.epoch_times))
        print(f"[Timing] Total training time: {total_training_sec:.3f} sec")
        print(f"[Timing] Average per-epoch time: {avg_epoch_sec:.3f} sec")

        timing_path = os.path.join(self.checkpoint_dir, "timing_train.txt")
        with open(timing_path, "w", encoding="utf-8") as f:
            f.write(f"Total training time (sec): {total_training_sec:.6f}\n")
            f.write(f"Average epoch time (sec): {avg_epoch_sec:.6f}\n")
            for i, t in enumerate(self.epoch_times, 1):
                f.write(f"epoch_{i:03d}: {t:.6f}\n")
        print(f"[INFO] Saved training timings to: {timing_path}")

    def plot_loss(self, train_loss_history):
        plt.figure()
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, "loss_graph.png"))
        plt.close()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("radius", type=int)
    parser.add_argument("dim", type=int)
    parser.add_argument("layer_hidden", type=int)
    parser.add_argument("batch_train", type=int)
    parser.add_argument("batch_test", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("iteration", type=int)
    parser.add_argument("checkpoint_dir", type=str)

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pth to resume from")
    parser.add_argument("--train_file", type=str, default="", help="Train data file path (.txt or .csv). Overrides default")
    parser.add_argument("--test_file", type=str, default="", help="Test data file path (.txt or .csv). Overrides default")
    parser.add_argument("--val_file", type=str, default="", help="Optional val data file path (.txt or .csv). If empty, split from train")
    parser.add_argument("--n_outputs", type=int, default=5, help="Number of peaks per SMILES to supervise")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"[INFO] Batch size (train): {args.batch_train}, Batch size (test): {args.batch_test}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp.set_seed(1234)

    train_file = args.train_file.strip() or None
    test_file = args.test_file.strip() or None
    val_file = args.val_file.strip() or None

    print("[INFO] Preprocessing the dataset...")
    dataset_train, dataset_val, dataset_test, N_fingerprints, atom_dict = pp.create_datasets(
        dataset_name=args.dataset,
        radius=args.radius,
        device=device,
        N_outputs=args.n_outputs,
        checkpoint_dir=args.checkpoint_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )

    noise_mode = os.getenv("NOISE_MODE", "hybrid_gamma")
    loss_mode = os.getenv("LOSS_MODE", "balanced")

    num_steps = 3  # keep consistent across runtime + checkpoint args

    model = DiffusionModel_Gaussian(
        N_fingerprints=N_fingerprints,
        dim=args.dim,
        num_layers=args.layer_hidden,
        latent_dim=128,
        device=device,
        noise_mode=noise_mode,
        loss_mode=loss_mode,
        mz_min=0.0,
        mz_max=500.0,
        multi_step_training=True,
        num_steps=num_steps,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    resume_path = (args.resume or "").strip()
    if resume_path:
        if os.path.exists(resume_path):
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)

            state_dict = ckpt.get("model_state_dict", {})
            for k in list(state_dict.keys()):
                if k.startswith("reverse_diffusion.dependency_proj."):
                    print(f"[WARN] dropping obsolete key from checkpoint: {k}")
                    state_dict.pop(k)

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print("[INFO] missing keys:", missing[:20])
            if unexpected:
                print("[INFO] unexpected keys:", unexpected[:20])

            opt_state = ckpt.get("optimizer_state_dict", None)
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)

            start_epoch = int(ckpt.get("epoch", 0))
            print(f"[INFO] Resume start_epoch set to: {start_epoch}")
        else:
            print(f"[WARN] Resume checkpoint not found: {resume_path}")
    else:
        print("[INFO] Resume disabled (fresh start). Use --resume <path> to resume.")

    model_args = {
        "N_fingerprints": N_fingerprints,
        "dim": args.dim,
        "num_layers": args.layer_hidden,
        "latent_dim": 128,
        "device": str(device),
        "feature_dim": 9,
        "noise_mode": noise_mode,
        "loss_mode": loss_mode,
        "mz_min": 0.0,
        "mz_max": 500.0,
        "multi_step_training": True,
        "num_steps": num_steps,
        "n_outputs": args.n_outputs,
    }

    if not resume_path:
        init_ckpt_path = os.path.join(args.checkpoint_dir, "model_checkpoint_initial.pth")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({
            "model_args": model_args,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
        }, init_ckpt_path)
        print(f"[INFO] Initial checkpoint saved at: {init_ckpt_path}")

    model.sample_metadata = {idx: {"smiles": sample[0]} for idx, sample in enumerate(dataset_train)}

    loss_file = os.path.join(args.checkpoint_dir, "loss_results.txt")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.checkpoint_dir,
        loss_file=loss_file,
        model_args=model_args,
        k_supervise=5,
        start_epoch=start_epoch,
    )
    trainer.train(dataset_train, dataset_val, args.iteration)
