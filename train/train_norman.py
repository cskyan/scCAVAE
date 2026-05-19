#!/usr/bin/env python3
"""
Norman Dataset Training Script for scCAVAE
Simple and efficient training with optimized parameters
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import scanpy as sc
import numpy as np
from datetime import datetime
from model import scCAVAE
import torch
from training_utils import load_dataset_for_training, setup_norman_data

# 启用 Tensor Cores 加速（A100 优化）
torch.set_float32_matmul_precision('high')
print("Tensor Cores precision set to 'high' for A100 optimization")

# GPU Configuration - use environment variable if set, otherwise default to GPU 0
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
print(f"Using GPU: {gpu_id}")

# Create save path first for logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = str(ROOT / "train" / "results" / "norman" / current_time)
os.makedirs(save_path, exist_ok=True)

# Setup logging - redirect stdout to both console and file
class Tee:
    """Redirect stdout to both console and file"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(f"{save_path}/training.log", 'w')
sys.stdout = Tee(sys.stdout, log_file)

print("=" * 80)
print("scCAVAE Training - Norman Dataset")
print("=" * 80)
print(f"Results will be saved to: {save_path}")
print(f"Training log: {save_path}/training.log")

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/5] Loading Norman dataset...")
# Use prepared data if available
adata, is_prepared = load_dataset_for_training('norman', use_prepared=True)
print(f"  Data shape: {adata.shape}")

# 读取重建损失类型，决定数据空间
RECON_LOSS = os.environ.get('RECON_LOSS', 'nb')
print(f"  Reconstruction loss: {RECON_LOSS}")
USE_LOG_SPACE = RECON_LOSS in ['mse', 'mse_sum', 'huber', 'gauss']

# ============================================================================
# 2. Setup AnnData
# ============================================================================
if not is_prepared:
    print("\n[2/5] Setting up AnnData...")
    adata = setup_norman_data(adata)
else:
    print("\n[2/5] Setting up AnnData (prepared data)...")

    import scipy.sparse as sp
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()

    if USE_LOG_SPACE:
        if sp.issparse(adata.X):
            adata.X = np.log1p(adata.X.toarray())
        else:
            adata.X = np.log1p(adata.X)
        print(f"  Log1p space: X range [{adata.X.min():.2f}, {adata.X.max():.2f}]")
    else:
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        print(f"  Raw count space: X range [{adata.X.min():.0f}, {adata.X.max():.0f}]")

    scCAVAE.setup_anndata(
        adata,
        perturbation_key='cond_harm',
        control_group='ctrl',
        dosage_key='dose_value',
        categorical_covariate_keys=['cell_type'],
        is_count_data=not USE_LOG_SPACE,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_cond',
        max_comb_len=2,
    )

# ============================================================================
# 3. Model Configuration
# ============================================================================
print("\n[3/5] Configuring model...")
model_params = {
    # Architecture - 基于tuner最优配置 (val_r2_mean=0.929)
    "n_latent": 128,
    "n_hidden_encoder": 384,
    "n_layers_encoder": 4,
    "n_hidden_decoder": 1024,
    "n_layers_decoder": 3,

    # Regularization
    "dropout_rate_encoder": 0.12,
    "dropout_rate_decoder": 0.08,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,

    # Loss and optimization
    "recon_loss": RECON_LOSS,
    "variational": True,

    # Perturbation network
    "doser_type": "none",
    "encoding_strategy": "combination_attention",
    "dose_aware": False,  # Norman 无剂量变化

    # Other
    "seed": 8206,
}

trainer_params = {
    # Learning rates
    "lr": 7.65e-6,
    "wd": 3.2e-6,

    # Training schedule
    "n_epochs_pretrain_ae": 30,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.1,
    "step_size_lr": 25,

    # Gradient clipping
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,

    # Adversarial training - 禁用
    "use_adversarial_training": False,

    # Contrastive learning - 组合扰动用 hierarchical_supcon
    "contrastive_loss_type": "hierarchical_supcon",
    "reg_contrastive": 2.5,
    "supcon_temperature": 0.067,
    "n_epochs_contrastive_warmup": 30,
}

print("\nModel parameters:")
for key, value in model_params.items():
    print(f"  {key}: {value}")

print("\nTrainer parameters:")
for key, value in trainer_params.items():
    print(f"  {key}: {value}")

# ============================================================================
# 4. Create Model and Train
# ============================================================================
print("\n[4/5] Creating model...")

model = scCAVAE(
    adata=adata,
    split_key='split_6',
    train_split='train',
    valid_split='valid',
    test_split='ood',
    **model_params,
)

print("\n[5/5] Starting training...")
print("=" * 80)

model.train(
    max_epochs=2000,
    use_gpu=True,
    batch_size=8192,
    plan_kwargs=trainer_params,
    early_stopping_patience=5,
    check_val_every_n_epoch=5,
    save_path=save_path,
    precision=16,  # AMP混合精度训练加速
    use_auto_optimization=True,
)

print("\n" + "=" * 80)
print("Training completed!")
print(f"Model saved to: {save_path}")
print("=" * 80)

# ============================================================================
# 5. Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("Evaluating model performance...")
print("=" * 80)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm

# Prepare for evaluation - replace with random control cells
print("\n[1/4] Preparing data for evaluation...")
adata.layers['X_true'] = adata.X.copy()
ctrl_adata = adata[adata.obs['cond_harm'] == 'ctrl'].copy()

# Replace X with random control cells for prediction
adata.X = ctrl_adata.X[np.random.choice(ctrl_adata.n_obs, size=adata.n_obs, replace=True), :]

print("\n[2/4] Generating predictions...")
model.predict(adata, batch_size=2048)
adata.layers['scCAVAE_pred'] = adata.obsm['scCAVAE_pred'].copy()

# Normalize and log-transform
print("\n[3/4] Normalizing and computing R² scores...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.normalize_total(adata, target_sum=1e4, layer='scCAVAE_pred')
sc.pp.log1p(adata, layer='scCAVAE_pred')

# Define evaluation metrics
n_top_degs = [10, 20, 50, None]  # None means all genes

results = defaultdict(list)
ctrl_adata = adata[adata.obs['cond_harm'] == 'ctrl'].copy()

for condition in tqdm(adata.obs['cond_harm'].unique(), desc="Evaluating conditions"):
    if condition != 'ctrl':
        cond_adata = adata[adata.obs['cond_harm'] == condition].copy()

        deg_cat = f'K562_{condition}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

        x_true = cond_adata.layers['counts'].toarray()
        x_pred = cond_adata.obsm['scCAVAE_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()

        for n_top_deg in n_top_degs:
            if n_top_deg is not None:
                degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
            else:
                degs = np.arange(adata.n_vars)
                n_top_deg = 'all'

            x_true_deg = x_true[:, degs]
            x_pred_deg = x_pred[:, degs]
            x_ctrl_deg = x_ctrl[:, degs]

            r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                       x_pred_deg.mean(0) - x_ctrl_deg.mean(0))

            results['condition'].append(condition)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)

df = pd.DataFrame(results)

print("\n[4/4] Saving evaluation results...")
results_csv_path = f"{save_path}/evaluation_results.csv"
df.to_csv(results_csv_path, index=False)
print(f"  Saved to: {results_csv_path}")

# Print summary
print("\n" + "=" * 80)
print("Evaluation Summary")
print("=" * 80)
print("\nOverall Performance:")
for n_deg in [10, 20, 50, 'all']:
    df_subset = df[df['n_top_deg'] == n_deg]
    if len(df_subset) > 0:
        print(f"\nTop {n_deg} DEGs:")
        print(f"  R² (mean):     {df_subset['r2_mean_deg'].mean():.4f} ± {df_subset['r2_mean_deg'].std():.4f}")
        print(f"  R² (mean LFC): {df_subset['r2_mean_lfc_deg'].mean():.4f} ± {df_subset['r2_mean_lfc_deg'].std():.4f}")

# Print OOD performance
print("\n" + "=" * 80)
print("OOD Test Set Performance:")
print("=" * 80)
ood_conditions = ['DUSP9+ETS2', 'CBL+CNN1']
ood_df = df[df['condition'].isin(ood_conditions)]
if len(ood_df) > 0:
    print("\nOOD Conditions: DUSP9+ETS2, CBL+CNN1")
    for n_deg in [10, 20, 50, 'all']:
        ood_subset = ood_df[ood_df['n_top_deg'] == n_deg]
        if len(ood_subset) > 0:
            print(f"\nTop {n_deg} DEGs:")
            print(f"  R² (mean):     {ood_subset['r2_mean_deg'].mean():.4f}")
            print(f"  R² (mean LFC): {ood_subset['r2_mean_lfc_deg'].mean():.4f}")

print("\n" + "=" * 80)
print("✓ Training and evaluation completed successfully!")
print("=" * 80)
print(f"Results location: {save_path}")
print(f"  - Model: model.pt")
print(f"  - Evaluation: evaluation_results.csv")
print(f"  - History: history.csv")
print("=" * 80)
