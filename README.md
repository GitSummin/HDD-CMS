# HDD-LIMA: Hybrid-noise & Dynamic-optimization Diffusion for Library-Independent Mass Spectrometry Analysis

HDD-LIMA is a structure-conditioned generative framework for synthesizing mass spectra from molecular structure representations (e.g., SMILES). The goal is to support spectrum generation and analysis in settings where reference libraries are incomplete or unavailable.

---

**Key components**
1. **Structure encoding**  
   Molecules are represented as graphs derived from SMILES and encoded into fixed-dimensional embeddings using a graph neural network (GNN). This embedding serves as the conditioning signal for generation.

2. **Conditional diffusion-based generation**  
   A diffusion-style denoising process is used to produce spectrum-related outputs conditioned on molecular embeddings. The model is trained to recover peak-related information through iterative refinement.

3. **Noise modeling tailored to MS characteristics**  
   The implementation includes a configurable noise formulation intended to better accommodate non-ideal statistical properties commonly observed in MS signals (e.g., asymmetry and heavy-tailed behavior in intensities).

4. **Training objective and optimization schedule**  
   The training procedure combines distribution-level alignment and peak-level refinement objectives, with an adjustable scheduling strategy that can shift emphasis over training.

**Evaluation protocol (code-level)**
Model evaluation in this repository is based on a top-*k* peak protocol and reports standard quantitative measures for:
- *m/z* localization quality (e.g., MAE/RMSE)
- intensity reconstruction quality (e.g., MAE/RMSE)
- spectrum-level agreement (e.g., cosine similarity and spectral angle mapper)

---

## 2. Repository Structure

```text
HDD-LIMA/
├─ data/
│  ├─ preprocess.py
│  └─ csv                      # internal datasets (not publicly released)
│
├─ model/
│  ├─ diffusion.py                # Structure-conditioned diffusion
│  ├─ reverse_diffusion.py        # ReverseDiffusionUNet backbone
│  ├─ gnn.py                      # GraphNeuralNetwork encoder
│  ├─ utils.py                    # FiLM, DependencyNormalization, and utilities
│  └─ __init__.py
│
├─ output/
│  └─ checkpoints/
│     └─ run_XX/
│        ├─ fingerprint_dict.pth
│        ├─ model_checkpoint_initial.pth
│        └─ model_checkpoint_epoch_*.pth
│
├─ train.py
├─ test.py
├─ train.bat
├─ test.bat
└─ README.md
````

---

## 3. Installation

### 3.1 Prerequisites

* OS: Windows (tested), Linux should also work with minor path adjustments
* Python: recommended 3.9–3.11
* GPU: optional but recommended for training (CUDA-enabled PyTorch)

### 3.2 Environment Setup (Conda recommended)

```bash
conda create -n HDDLIMA python=3.10 -y
conda activate HDDLIMA
pip install -r requirements.txt
```

---

## 4. Data Policy and Expected Formats

### 4.1 Data Availability

This repository **does not include** the raw or processed MS datasets used in the paper. The datasets are internally curated and can be shared only upon reasonable request (project-specific policy).

### 4.2 Dataset Placement

Place your CSV files under:

```text
data/
  your_train_dataset.csv
  your_val_dataset.csv
  your_test_dataset.csv
```

At minimum, evaluation via `test.py` requires that the CSV contains a **SMILES** column. If you want category-stratified reporting, include a **label** column.

**Recommended columns for evaluation CSV:**

* `smiles` (required): SMILES string
* `label` (optional): scenario/category label (e.g., Fire / Poisoning / Terrorism / Polyester / Nylon)

Any additional columns required by your preprocessing pipeline (e.g., peak arrays, metadata, auxiliary features) should be described in your internal schema documentation.

---

## 5. Training

### 5.1 Running training

Training is orchestrated by `train.py`. For Windows, `train.bat` is the reference entry point and typically encodes the exact arguments used for internal experiments.

```bash
# Option A (recommended on Windows): use the provided batch script
train.bat

# Option B: run directly (arguments depend on your implementation in train.py)
python train.py <...your arguments...>
```

### 5.2 Outputs

A typical run creates:

* `output/checkpoints/run_XX/fingerprint_dict.pth`
  Dictionary mapping fingerprint representation to indices (used at evaluation time).
* `output/checkpoints/run_XX/model_checkpoint_*.pth`
  Model weights and `model_args` (used to restore architecture at test time).

---

## 6. Evaluation

Evaluation is implemented in `test.py`. The script loads:

1. A model checkpoint (`.pth`)
2. The fingerprint dictionary (`fingerprint_dict.pth`) from the corresponding training run directory
3. A test CSV file

### 6.1 Command

```bash
python test.py <checkpoint_path> <output_dir> ^
  --test_file <csv_path> ^
  --checkpoint_dir <train_run_dir_containing_fingerprint_dict> ^
  --n_outputs 5 ^
  --radius 1
```

**Arguments**

* `<checkpoint_path>`: path to a trained model checkpoint (e.g., `output/checkpoints/run_01/model_checkpoint_epoch_XXX.pth`)
* `<output_dir>`: directory where predictions and metrics will be saved
* `--test_file`: test CSV (must include at least `smiles`; `label` optional)
* `--checkpoint_dir`: directory containing `fingerprint_dict.pth`
* `--n_outputs`: number of output peaks/slots expected by the preprocessing pipeline
* `--radius`: fingerprint radius used in preprocessing

### 6.2 Evaluation Outputs

`test.py` produces:

* `predictions.xlsx`

  * `predictions` sheet: per-SMILES Top-*k* rows including predicted and ground-truth peaks
  * `metrics_overall`, `metrics_by_category`, `metrics_by_rank`, `metrics_cat_rank`
* `metrics.csv`

  * merged metrics blocks for convenient downstream plotting/aggregation

### 6.3 Metrics

Reported metrics include:

* *m/z*: MAE, RMSE, Cosine similarity, SAM (and optionally PPM-MAE)
* intensity: MAE, RMSE, Cosine similarity, SAM

---

## License

- **Source code**: Apache-2.0 License (see `LICENSE`)
- **Datasets**: Not publicly available. Access may be granted upon reasonable request for peer review or non-commercial academic research. Redistribution is not permitted without explicit written permission.

