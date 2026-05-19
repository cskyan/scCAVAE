#!/usr/bin/env python3
"""
Kang Dataset Training Script for scCAVAE
Simple and efficient training with optimized parameters
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import scanpy as sc
from datetime import datetime
from model import scCAVAE
import torch
from training_utils import load_dataset_for_training, setup_kang_data
import numpy as np
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
save_path = str(ROOT / "train" / "results" / "kang" / current_time)
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
print("scCAVAE Training - Kang Dataset")
print("=" * 80)
print(f"Results will be saved to: {save_path}")
print(f"Training log: {save_path}/training.log")

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/5] Loading Kang dataset...")
# Use prepared data if available
adata, is_prepared = load_dataset_for_training('kang', use_prepared=True)
print(f"  Data shape: {adata.shape}")

# 提前读取重建损失类型，决定数据空间处理
RECON_LOSS = os.environ.get('RECON_LOSS', 'nb')
print(f"  Reconstruction loss: {RECON_LOSS}")

# 根据 loss 类型决定数据空间：
#   MSE/Huber/Gauss: log1p 空间（压缩动态范围，各基因贡献均匀）
#   NB/ZINB: 原始计数空间（模型内部自行处理 log1p 给 encoder，loss 在原始空间算）
USE_LOG_SPACE = RECON_LOSS in ['mse', 'mse_sum', 'huber', 'gauss']

# ============================================================================
# 2. Setup AnnData
# ============================================================================
if not is_prepared:
    print("\n[2/5] Setting up AnnData...")
    adata = setup_kang_data(adata)
else:
    print("\n[2/5] Setting up AnnData (prepared data)...")

    import scipy.sparse as sp
    # 始终保存原始计数到 layers['counts']
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()

    if USE_LOG_SPACE:
        # MSE/Huber/Gauss: 转换到 log1p 空间训练
        if sp.issparse(adata.X):
            adata.X = np.log1p(adata.X.toarray())
        else:
            adata.X = np.log1p(adata.X)
        print(f"  Log1p space: X range [{adata.X.min():.2f}, {adata.X.max():.2f}]")
    else:
        # NB/ZINB: 保持原始计数，模型内部处理
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
        print(f"  Raw count space: X range [{adata.X.min():.0f}, {adata.X.max():.0f}]")

    scCAVAE.setup_anndata(
        adata,
        perturbation_key='condition',
        control_group='ctrl',
        dosage_key='dose',
        categorical_covariate_keys=['cell_type'],
        is_count_data=not USE_LOG_SPACE,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_cond',
        max_comb_len=1,
    )

# ============================================================================
# 3. Model Configuration
# ============================================================================
print("\n[3/5] Configuring model...")

# 验证损失类型
valid_losses = ['mse', 'nb', 'zinb', 'gauss', 'huber']
if RECON_LOSS not in valid_losses:
    print(f"⚠️  Warning: Invalid loss type '{RECON_LOSS}', using 'mse' instead")
    RECON_LOSS = 'mse'

model_params = {
    # Architecture - 基于tuner最优trial (val_r2_mean=0.974)
    "n_latent": 64,
    "n_hidden_encoder": 256,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 768,
    "n_layers_decoder": 2,

    # Regularization
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.0,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,

    # Loss and optimization
    "recon_loss": RECON_LOSS,
    "variational": True,  # tuner最优使用variational

    # Perturbation network
    "doser_type": "logsigm",
    "encoding_strategy": "combination_attention",
    "dose_aware": False,  # Kang无剂量变化

    # Other
    "seed": 6977,
}

trainer_params = {
    # Learning rates - tuner最优配置
    "lr": 0.0002,
    "wd": 2e-07,

    # Training schedule
    "n_epochs_pretrain_ae": 30,
    "step_size_lr": 25,

    # Gradient clipping
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,

    # Contrastive learning - tuner最优使用supcon
    "contrastive_loss_type": "supcon",
    "reg_contrastive": 0.6,
    "supcon_temperature": 0.1,
    "n_epochs_contrastive_warmup": 30,

    # Adversarial training - 禁用
    "use_adversarial_training": False,
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
    split_key='split_B',
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
    batch_size=1024,  # tuner最优512，但配合num_workers优化用1024兼顾速度
    plan_kwargs=trainer_params,
    early_stopping_patience=10,
    check_val_every_n_epoch=5,
    save_path=save_path,
    precision=16,  # AMP混合精度训练加速
    use_auto_optimization=True,  # 非对抗训练使用auto计划，兼容AMP
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
import scipy.sparse as sp
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm

# Generate predictions
print("\n[1/3] Generating predictions...")
model.predict(adata, batch_size=2048)

# 评估空间判断：与训练空间一致
# NB/ZINB: 预测在原始尺度 → 需要 log1p
# MSE/Huber/Gauss: 预测已在 log1p 空间 → 不需要
recon_loss_type = model.module.recon_loss
pred_needs_log = not USE_LOG_SPACE
print(f"\nDetected reconstruction loss: {recon_loss_type}")
print(f"  Prediction space: {'raw counts → apply log1p' if pred_needs_log else 'log1p (no transform needed)'}")

# Define evaluation metrics
n_top_degs = [10, 20, 50, None]  # None means all genes

def safe_log1p(x):
    """安全地对数据应用log1p，处理稀疏矩阵"""
    if sp.issparse(x):
        return np.log1p(x.toarray())
    else:
        return np.log1p(x)

print("\n[2/3] Computing R² scores for different DEG sets...")
results = defaultdict(list)

for cat in tqdm(adata.obs['cov_cond'].unique(), desc="Evaluating conditions"):
    if 'ctrl' not in cat:
        cov, condition = cat.split('_')
        # 使用obs_names来进行索引
        cat_obs_names = adata.obs[adata.obs['cov_cond'] == cat].index
        ctrl_obs_names = adata.obs[adata.obs['cov_cond'] == f'{cov}_ctrl'].index
        cat_adata = adata[cat_obs_names].copy()
        ctrl_adata = adata[ctrl_obs_names].copy()

        deg_cat = f'{cat}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

        x_true = cat_adata.layers['counts']  # 原始计数
        x_pred = cat_adata.obsm['scCAVAE_pred']
        x_ctrl = ctrl_adata.layers['counts']  # 原始计数

        # 真值统一转到 log1p 空间
        x_true = safe_log1p(x_true)
        x_ctrl = safe_log1p(x_ctrl)

        # 预测值：NB/ZINB 输出原始尺度需要 log1p；MSE 输出已在 log 空间
        if pred_needs_log:
            x_pred = safe_log1p(x_pred)
        else:
            if sp.issparse(x_pred):
                x_pred = x_pred.toarray()

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
            r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))

            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                       x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0),
                                      x_pred_deg.var(0) - x_ctrl_deg.var(0))

            results['condition'].append(condition)
            results['cell_type'].append(cov)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_var_deg'].append(r2_var_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
            results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

df = pd.DataFrame(results)

print("\n[3/3] Saving evaluation results...")
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
        print(f"  R² (var):      {df_subset['r2_var_deg'].mean():.4f} ± {df_subset['r2_var_deg'].std():.4f}")
        print(f"  R² (mean LFC): {df_subset['r2_mean_lfc_deg'].mean():.4f} ± {df_subset['r2_mean_lfc_deg'].std():.4f}")
        print(f"  R² (var LFC):  {df_subset['r2_var_lfc_deg'].mean():.4f} ± {df_subset['r2_var_lfc_deg'].std():.4f}")

# Print OOD performance (stimulated condition)
print("\n" + "=" * 80)
print("OOD Test Set Performance (stimulated):")
print("=" * 80)
ood_df = df[df['condition'] == 'stimulated']
if len(ood_df) > 0:
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
