import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import preprocess as pp
from model.diffusion import DiffusionModel_Gaussian


APPLY_MZ_CLAMP = False
TOPK = 3
NORMALIZE_TOPK_TO_TOP1 = True

PRINT_PREDICTED_PEAKS = True
PRINT_MAX_SHOW = 15
ZERO_FLOOR = float(os.getenv("ZERO_FLOOR", "0.001")) 

def replace_exact_zeros(arr: np.ndarray, floor: float) -> np.ndarray:
    a = np.asarray(arr, float).copy()
    a[a == 0.0] = floor
    return a


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred, eps=1e-9):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    denom = np.abs(y_true) + eps
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cosine_similarity(y_true, y_pred, eps=1e-9):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    dot = np.sum(y_true * y_pred)
    norm = (np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + eps
    return float(dot / norm)


def spectral_angle_mapper(y_true, y_pred, eps=1e-9):
    cos_sim = cosine_similarity(y_true, y_pred, eps=eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle_rad = np.arccos(cos_sim)
    return float(np.degrees(angle_rad))


def ppm_mae(y_true, y_pred, eps=1e-9):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    ppm = 1e6 * np.abs(y_pred - y_true) / (np.abs(y_true) + eps)
    return float(np.nanmean(ppm))


def set_zero_dependency_like(fingerprints: torch.Tensor) -> torch.Tensor:
    B = fingerprints.size(0)
    device = fingerprints.device
    return torch.zeros(B, 1, device=device, dtype=torch.float32)


def tensor_row_to_np1d(t: torch.Tensor, b: int = 0) -> np.ndarray:
    if not isinstance(t, torch.Tensor):
        arr = np.asarray(t)
        return arr.reshape(-1)
    if t.dim() >= 2:
        t = t[b]
    return t.detach().cpu().reshape(-1).numpy()


def _paired_filter(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _metric_dict(y_true, y_pred, is_mz=False) -> dict:
    y_true, y_pred = _paired_filter(y_true, y_pred)
    if y_true.size == 0:
        return {
            "count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE(%)": np.nan,
            "CosSim": np.nan,
            "SAM": np.nan,
            "R2": np.nan,
            "PPM_MAE": np.nan if is_mz else None,
        }
    out = {
        "count": int(y_true.size),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "CosSim": cosine_similarity(y_true, y_pred),
        "SAM": spectral_angle_mapper(y_true, y_pred),
        "R2": r2_score(y_true, y_pred) if y_true.size >= 2 else np.nan,
    }
    if is_mz:
        out["PPM_MAE"] = ppm_mae(y_true, y_pred)
    return out


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> DiffusionModel_Gaussian:
    print(f"[INFO] Loading checkpoint: {ckpt_path} on {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    margs = ckpt.get("model_args", {})

    model = DiffusionModel_Gaussian(
        N_fingerprints=margs.get("N_fingerprints"),
        dim=margs.get("dim", 50),
        num_layers=margs.get("num_layers", 6),
        latent_dim=margs.get("latent_dim", 128),
        device=device,
        noise_mode=margs.get("noise_mode", "hybrid_gamma"),
        loss_mode=margs.get("loss_mode", "balanced"),
        mz_min=margs.get("mz_min", 0.0),
        mz_max=margs.get("mz_max", 500.0),
        multi_step_training=margs.get("multi_step_training", True),
        num_steps=margs.get("num_steps", 3),
        normalize_pred_intensity=True,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print("  missing keys (truncated):", missing[:10])
        if unexpected:
            print("  unexpected keys (truncated):", unexpected[:10])

    model.eval()
    print("[INFO] Model loaded and set to eval mode.")
    return model


def compute_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _one_scope(sub: pd.DataFrame, scope_cols: dict) -> pd.DataFrame:
        mz_true = pd.to_numeric(sub["Actual m/z"], errors="coerce").values
        mz_pred = pd.to_numeric(sub["Pred m/z"], errors="coerce").values
        m_mz = _metric_dict(mz_true, mz_pred, is_mz=True)

        it_true = pd.to_numeric(sub["Actual Intensity"], errors="coerce").values
        it_pred = pd.to_numeric(sub["Pred Intensity"], errors="coerce").values
        m_i = _metric_dict(it_true, it_pred, is_mz=False)

        row = {}
        row.update(scope_cols)
        row.update({
            "mz_count": m_mz["count"],
            "mz_MAE": m_mz["MAE"],
            "mz_RMSE": m_mz["RMSE"],
            "mz_MAPE(%)": m_mz["MAPE(%)"],
            "mz_CosSim": m_mz["CosSim"],
            "mz_SAM": m_mz["SAM"],
        })
        row.update({
            "I_count": m_i["count"],
            "I_MAE": m_i["MAE"],
            "I_RMSE": m_i["RMSE"],
            "I_MAPE(%)": m_i["MAPE(%)"],
            "I_CosSim": m_i["CosSim"],
            "I_SAM": m_i["SAM"],
        })
        return pd.DataFrame([row])

    metrics_overall = _one_scope(df, {"Scope": "Overall"})

    rows = []
    for cat, sub in df.groupby("Category", dropna=False):
        rows.append(_one_scope(sub, {"Scope": "ByCategory", "Category": cat}))
    metrics_by_category = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows = []
    for label, sub in df.groupby("Label", dropna=False):
        rows.append(_one_scope(sub, {"Scope": "ByRank", "Label": label}))
    metrics_by_rank = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    rows = []
    for (cat, label), sub in df.groupby(["Category", "Label"], dropna=False):
        rows.append(_one_scope(sub, {"Scope": "ByCat×Rank", "Category": cat, "Label": label}))
    metrics_cat_rank = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return metrics_overall, metrics_by_category, metrics_by_rank, metrics_cat_rank


def evaluate_and_save(records, output_excel, category_order):
    df = pd.DataFrame(records)

    def _round_metrics(mdf: pd.DataFrame) -> pd.DataFrame:
        if mdf is None or mdf.empty:
            return mdf
        mdf = mdf.copy()

        numeric_cols = [c for c in mdf.columns if pd.api.types.is_numeric_dtype(mdf[c])]
        count_cols = [c for c in numeric_cols if c.endswith("_count")]
        value_cols = [c for c in numeric_cols if c not in count_cols]

        if value_cols:
            mdf[value_cols] = mdf[value_cols].round(2)

        for c in count_cols:
            mdf[c] = mdf[c].astype("Int64")

        return mdf

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")

        m_overall, m_by_cat, m_by_rank, m_cat_rank = compute_blocks(df)

        if not m_by_cat.empty and "Category" in m_by_cat.columns:
            m_by_cat["__cat_order"] = m_by_cat["Category"].apply(
                lambda x: category_order.index(x) if x in category_order else len(category_order)
            )
            m_by_cat = m_by_cat.sort_values(["__cat_order"]).drop(columns="__cat_order")

        if not m_cat_rank.empty and "Category" in m_cat_rank.columns:
            m_cat_rank["__cat_order"] = m_cat_rank["Category"].apply(
                lambda x: category_order.index(x) if x in category_order else len(category_order)
            )
            rank_order = {"Top1": 1, "Top2": 2, "Top3": 3}
            m_cat_rank["__rank_order"] = m_cat_rank["Label"].map(rank_order).fillna(99)
            m_cat_rank = m_cat_rank.sort_values(["__cat_order", "__rank_order"]).drop(
                columns=["__cat_order", "__rank_order"]
            )

        m_overall_r = _round_metrics(m_overall)
        m_by_cat_r = _round_metrics(m_by_cat)
        m_by_rank_r = _round_metrics(m_by_rank)
        m_cat_rank_r = _round_metrics(m_cat_rank)

        m_overall_r.to_excel(writer, index=False, sheet_name="metrics_overall")
        if not m_by_cat_r.empty:
            m_by_cat_r.to_excel(writer, index=False, sheet_name="metrics_by_category")
        if not m_by_rank_r.empty:
            m_by_rank_r.to_excel(writer, index=False, sheet_name="metrics_by_rank")
        if not m_cat_rank_r.empty:
            m_cat_rank_r.to_excel(writer, index=False, sheet_name="metrics_cat_rank")

    base_dir = os.path.dirname(output_excel)
    metrics_csv = os.path.join(base_dir, "metrics.csv")

    parts = []
    if not m_overall.empty:
        t = _round_metrics(m_overall)
        t.insert(0, "Block", "overall")
        parts.append(t)
    if not m_by_cat.empty:
        t = _round_metrics(m_by_cat)
        t.insert(0, "Block", "by_category")
        parts.append(t)
    if not m_by_rank.empty:
        t = _round_metrics(m_by_rank)
        t.insert(0, "Block", "by_rank")
        parts.append(t)
    if not m_cat_rank.empty:
        t = _round_metrics(m_cat_rank)
        t.insert(0, "Block", "by_cat_rank")
        parts.append(t)

    if parts:
        pd.concat(parts, ignore_index=True).to_csv(metrics_csv, index=False, encoding="utf-8-sig")


def _mean_over_steps(x):
    if isinstance(x, list):
        return torch.stack(x, dim=0).mean(dim=0)
    return x


def load_category_map_from_csv(csv_path: str) -> Dict[str, str]:
    """
    If CSV has a 'label' column, use it as Category in outputs.
    Otherwise Category is 'Unknown'.
    """
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cols = [c.strip().lower() for c in df.columns]
    if "smiles" not in cols:
        return mapping
    if "label" not in cols:
        return mapping

    smi_col = df.columns[cols.index("smiles")]
    lab_col = df.columns[cols.index("label")]
    for _, r in df[[smi_col, lab_col]].dropna().iterrows():
        mapping[str(r[smi_col])] = str(r[lab_col])
    return mapping


if __name__ == "__main__":
    # New usage:
    #   python test.py <checkpoint_path> <output_dir> --test_file <csv_path> --checkpoint_dir <train_run_dir> [--n_outputs 5] [--radius 1]
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python test.py <checkpoint_path> <output_dir> --test_file <csv_path> --checkpoint_dir <dir> [--n_outputs 5] [--radius 1]"
        )

    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # simple manual arg parse
    def _get_arg(name: str, default=None):
        if name in sys.argv:
            i = sys.argv.index(name)
            if i + 1 < len(sys.argv):
                return sys.argv[i + 1]
        return default

    test_file = _get_arg("--test_file", "")
    checkpoint_dir = _get_arg("--checkpoint_dir", "")
    n_outputs = int(_get_arg("--n_outputs", "5"))
    radius = int(_get_arg("--radius", "1"))

    if not test_file:
        raise ValueError("Missing required arg: --test_file <csv_path>")
    if not checkpoint_dir:
        raise ValueError("Missing required arg: --checkpoint_dir <dir> (must contain fingerprint_dict.pth)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pp.set_seed(1234)

    # IMPORTANT: Do NOT call create_datasets() here.
    # Build test dataset from CSV using saved fingerprint_dict.
    dataset_test, N_fingerprints = pp.create_external_test_dataset(
        test_path=test_file,
        radius=radius,
        device=device,
        N_outputs=n_outputs,
        checkpoint_dir=checkpoint_dir,
    )

    smiles_category: Dict[str, str] = load_category_map_from_csv(test_file)

    model = load_model_from_checkpoint(checkpoint_path, device)
    output_excel = os.path.join(output_dir, "predictions.xlsx")

    records = []
    seen_smiles = set()
    printed_smiles = set()

    category_order = ["Fire", "Poisoning", "Terrorism", "Polyester", "Nylon", "Unknown"]

    for smiles, fingerprints, adjacencies, mol_sizes, mz, intensity, dependency, features in dataset_test:
        fingerprints = fingerprints.to(device)
        adjacencies = adjacencies.to(device)
        mol_sizes = mol_sizes.to(device)
        features = features.to(device)
        mz = mz.to(device)
        intensity = intensity.to(device)

        B = fingerprints.size(0)
        dep_zero = set_zero_dependency_like(fingerprints)

        with torch.no_grad():
            t_full = torch.full((B,), 1, dtype=torch.long, device=device)
            pred_mz_list_all, pred_I_list_all, *_ = model(
                fingerprints, adjacencies, mol_sizes, dep_zero, features, t_full
            )

        pred_mz_all = _mean_over_steps(pred_mz_list_all).clamp(
            min=model.mz_min - 1.0, max=model.mz_max + 1.0
        )
        pred_I_all = _mean_over_steps(pred_I_list_all).clamp(0, 1)

        if APPLY_MZ_CLAMP:
            pred_mz_all = torch.clamp(pred_mz_all, min=model.mz_min, max=model.mz_max)

        if NORMALIZE_TOPK_TO_TOP1:
            top1 = pred_I_all.max(dim=1, keepdim=True).values.clamp_min(1e-6)
            pred_I_all = (pred_I_all / top1).clamp(0.0, 1.0)

        for b in range(B):
            mz_b = tensor_row_to_np1d(mz, b)
            I_b = tensor_row_to_np1d(intensity, b)
            order_true = np.argsort(-I_b)
            tgt_topk_mz = mz_b[order_true][:TOPK]
            tgt_topk_I = I_b[order_true][:TOPK]

            mz_all_b = pred_mz_all[b].detach().cpu().numpy().reshape(-1)
            I_all_b = pred_I_all[b].detach().cpu().numpy().reshape(-1)

            order_all = np.lexsort((mz_all_b, -I_all_b))
            mz_all_b, I_all_b = mz_all_b[order_all], I_all_b[order_all]

            top_mz_b = mz_all_b[:TOPK]
            top_I_b = I_all_b[:TOPK]
            top_I_b = replace_exact_zeros(top_I_b, ZERO_FLOOR)

            if isinstance(smiles, str):
                smiles_b = smiles
            elif isinstance(smiles, (list, tuple)):
                smiles_b = smiles[b]
            elif torch.is_tensor(smiles):
                if smiles.dim() == 0:
                    smiles_b = str(smiles.item())
                else:
                    smiles_b = str(smiles[b].item())
            else:
                try:
                    smiles_b = smiles[b]
                except Exception:
                    smiles_b = str(smiles)

            if PRINT_PREDICTED_PEAKS and (smiles_b not in printed_smiles):
                cat = smiles_category.get(smiles_b, "Unknown")
                print("\n" + "-" * 72)
                print(f"[SMILES] {smiles_b}   |   Category: {cat}")
                for i in range(min(PRINT_MAX_SHOW, len(top_mz_b))):
                    pred_mz_i = top_mz_b[i] if i < len(top_mz_b) else np.nan
                    pred_I_i = top_I_b[i] if i < len(top_I_b) else np.nan
                    true_mz_i = tgt_topk_mz[i] if i < len(tgt_topk_mz) else np.nan
                    true_I_i = tgt_topk_I[i] if i < len(tgt_topk_I) else np.nan
                    print(
                        f"  [Top-{i+1}] Pred m/z={pred_mz_i:8.3f}, I={pred_I_i:.3f} | "
                        f"True m/z={true_mz_i:8.3f}, I={true_I_i:.3f}"
                    )
                printed_smiles.add(smiles_b)

            if smiles_b in seen_smiles:
                continue
            seen_smiles.add(smiles_b)

            for rank in range(TOPK):
                records.append({
                    "SMILES": smiles_b,
                    "Label": f"Top{rank+1}",
                    "Category": smiles_category.get(smiles_b, "Unknown"),
                    "Actual m/z": float(tgt_topk_mz[rank]) if rank < len(tgt_topk_mz) else np.nan,
                    "Actual Intensity": float(tgt_topk_I[rank]) if rank < len(tgt_topk_I) else np.nan,
                    "Pred m/z": float(top_mz_b[rank]) if rank < len(top_mz_b) else np.nan,
                    "Pred Intensity": float(top_I_b[rank]) if rank < len(top_I_b) else np.nan,
                })

    evaluate_and_save(records, output_excel, category_order)
    print(f"[INFO] Saved predictions to: {output_excel}")
