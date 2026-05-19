"""
Sciplex Dataset Training Script for scCAVAE
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
from training_utils import load_dataset_for_training, setup_sciplex_data
import torch

torch.set_float32_matmul_precision('high')  
print("Tensor Cores precision set to 'high' for A100 optimization")

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
print(f"Using GPU: {gpu_id}")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = str(ROOT / "train" / "results" / "sciplex" / current_time)
os.makedirs(save_path, exist_ok=True)

class Tee:
    
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
print("scCAVAE Training - Sciplex Dataset")
print("=" * 80)
print(f"Results will be saved to: {save_path}")
print(f"Training log: {save_path}/training.log")

print("\n[1/5] Loading Sciplex dataset...")

adata, is_prepared = load_dataset_for_training('sciplex', use_prepared=True, verbose=True)

if not is_prepared:
    adata = setup_sciplex_data(adata=None, verbose=True)

if is_prepared:
    print("\n[2/5] Setting up AnnData (prepared data)...")
    scCAVAE.setup_anndata(
        adata,
        perturbation_key='perturbation',
        control_group='Vehicle',
        dosage_key='dose',
        categorical_covariate_keys=['cell_type', 'replicate'],
        is_count_data=False,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_pert_name',
        max_comb_len=1,
    )
else:
    print("\n[2/5] Setup AnnData - COMPLETED")

print("\n[3/5] Configuring model...")

RECON_LOSS = os.environ.get('RECON_LOSS', 'mse')
print(f"🔧 Using reconstruction loss: {RECON_LOSS}")

valid_losses = ['mse', 'nb', 'zinb', 'gauss', 'huber']
if RECON_LOSS not in valid_losses:
    print(f"⚠️  Warning: Invalid loss type '{RECON_LOSS}', using 'mse' instead")
    RECON_LOSS = 'mse'

model_params = {
    
    "n_latent": 128,
    "n_hidden_encoder": 512,
    "n_layers_encoder": 3,
    "n_hidden_decoder": 768,
    "n_layers_decoder": 3,

    
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.1,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,

    
    "recon_loss": RECON_LOSS,
    "variational": False,

    
    "doser_type": "logsigm",
    "encoding_strategy": "combination_attention",
    "dose_aware": True,  

    
    "seed": 42,
}

trainer_params = {
    
    "lr": 0.0003,
    "wd": 1e-6,

    
    "n_epochs_pretrain_ae": 30,
    "step_size_lr": 20,

    
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,

    
    "contrastive_loss_type": "hierarchical_supcon",
    "reg_contrastive": 0.8,
    "supcon_temperature": 0.1,
    "n_epochs_contrastive_warmup": 30,

    
    "use_adversarial_training": False,
}

print("\nModel parameters:")
for key, value in model_params.items():
    print(f"  {key}: {value}")

print("\nTrainer parameters:")
for key, value in trainer_params.items():
    print(f"  {key}: {value}")

print("\n[4/5] Creating model...")

if 'split' not in adata.obs.columns:
    adata.obs['split'] = np.random.choice(['train', 'valid', 'test'],
                                          size=adata.n_obs,
                                          p=[0.7, 0.15, 0.15])
    print(f"  Created split distribution: {adata.obs['split'].value_counts().to_dict()}")
else:
    print(f"  Using existing split: {adata.obs['split'].value_counts().to_dict()}")

model = scCAVAE(
    adata=adata,
    split_key='split',
    train_split='train',
    valid_split='test',  
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
    early_stopping_patience=10,
    check_val_every_n_epoch=5,
    save_path=save_path,
    precision=16,  
    use_auto_optimization=True,
)

print("\n" + "=" * 80)
print("Training completed!")
print(f"Model saved to: {save_path}")
print("=" * 80)

print("\n" + "=" * 80)
print("Evaluating model performance...")
print("=" * 80)

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from tqdm import tqdm

print("\n[1/3] Generating predictions on OOD test set...")
if 'split' in adata.obs.columns:
    test_adata = adata[adata.obs['split'] == 'ood'].copy()  
    print(f"  OOD test set size: {len(test_adata)} cells")
else:
    test_adata = adata.copy()
    print(f"  Using full dataset: {len(test_adata)} cells")

model.predict(test_adata, batch_size=2048)

recon_loss_type = model.module.recon_loss
print(f"\n🔍 Detected reconstruction loss: {recon_loss_type}")

print("✓ Sciplex data is already log-transformed, using predictions as-is")
apply_log_transform = False

n_top_degs = [10, 20, 50, None]  

print("\n[2/3] Computing R² scores for different DEG sets...")
results = defaultdict(list)

if 'cov_drug_dose' in adata.obs.columns and 'rank_genes_groups_cov' in adata.uns:
    
    
    ctrl_adata = adata[adata.obs['perturbation'] == 'Vehicle'].copy()

    for cat in tqdm(test_adata.obs['cov_drug_dose'].unique(), desc="Evaluating conditions"):
        if 'Vehicle' not in cat:
            cat_adata = test_adata[test_adata.obs['cov_drug_dose'] == cat].copy()
            
            deg_cat = f'{cat}'
            if deg_cat not in adata.uns['rank_genes_groups_cov']:
                continue
                
            deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]
            
            x_true = cat_adata.X.toarray() if hasattr(cat_adata.X, 'toarray') else cat_adata.X
            x_pred = cat_adata.obsm['scCAVAE_pred']
            x_ctrl = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, 'toarray') else ctrl_adata.X

            
            if apply_log_transform:
                x_true = np.log1p(x_true)
                x_pred = np.log1p(x_pred)
                x_ctrl = np.log1p(x_ctrl)
            
            
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
                
                mse_mean_deg = mean_squared_error(x_true_deg.mean(0), x_pred_deg.mean(0))
                mse_mean_lfc_deg = mean_squared_error(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                                      x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
                
                
                parts = cat.split('_')
                if len(parts) >= 3:
                    cov, cond, dose = parts[0], parts[1], parts[2]
                else:
                    cov, cond, dose = 'unknown', cat, 'unknown'
                
                results['cell_type'].append(cov)
                results['condition'].append(cond)
                results['dose'].append(dose)
                results['n_top_deg'].append(n_top_deg)
                results['r2_mean_deg'].append(r2_mean_deg)
                results['r2_var_deg'].append(r2_var_deg)
                results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
                results['r2_var_lfc_deg'].append(r2_var_lfc_deg)
                results['mse_mean_deg'].append(mse_mean_deg)
                results['mse_mean_lfc_deg'].append(mse_mean_lfc_deg)
else:
    
    print("  Warning: Missing 'cov_drug_dose' or DEG info, using simplified evaluation")

    
    ctrl_adata = adata[adata.obs['perturbation'] == 'Vehicle'].copy()

    for perturbation in tqdm(test_adata.obs['perturbation'].unique(), desc="Evaluating perturbations"):
        if perturbation != 'Vehicle':
            cond_adata = test_adata[test_adata.obs['perturbation'] == perturbation].copy()
            
            x_true = cond_adata.X.toarray() if hasattr(cond_adata.X, 'toarray') else cond_adata.X
            x_pred = cond_adata.obsm['scCAVAE_pred']
            x_ctrl = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, 'toarray') else ctrl_adata.X

            
            if apply_log_transform:
                x_true = np.log1p(x_true)
                x_pred = np.log1p(x_pred)
                x_ctrl = np.log1p(x_ctrl)
            
            
            
            r2_mean = r2_score(x_true.mean(0), x_pred.mean(0))
            r2_var = r2_score(x_true.var(0), x_pred.var(0))
            r2_mean_lfc = r2_score(x_true.mean(0) - x_ctrl.mean(0),
                                   x_pred.mean(0) - x_ctrl.mean(0))

            results['perturbation'].append(perturbation)
            results['n_top_deg'].append('all')
            results['r2_mean_deg'].append(r2_mean)
            results['r2_var_deg'].append(r2_var)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc)

df = pd.DataFrame(results)

print("\n[3/3] Saving evaluation results...")
results_csv_path = f"{save_path}/evaluation_results.csv"
df.to_csv(results_csv_path, index=False)
print(f"  Saved to: {results_csv_path}")

print("\n" + "=" * 80)
print("Evaluation Summary")
print("=" * 80)

if len(df) > 0:
    print("\nOverall Performance:")
    for n_deg in [10, 20, 50, 'all']:
        df_subset = df[df['n_top_deg'] == n_deg]
        if len(df_subset) > 0:
            print(f"\nTop {n_deg} DEGs:")
            print(f"  R² (mean):     {df_subset['r2_mean_deg'].mean():.4f} ± {df_subset['r2_mean_deg'].std():.4f}")
            if 'r2_var_deg' in df_subset.columns:
                print(f"  R² (var):      {df_subset['r2_var_deg'].mean():.4f} ± {df_subset['r2_var_deg'].std():.4f}")
            if 'r2_mean_lfc_deg' in df_subset.columns:
                print(f"  R² (mean LFC): {df_subset['r2_mean_lfc_deg'].mean():.4f} ± {df_subset['r2_mean_lfc_deg'].std():.4f}")
            if 'mse_mean_deg' in df_subset.columns:
                print(f"  MSE (mean):    {df_subset['mse_mean_deg'].mean():.4f} ± {df_subset['mse_mean_deg'].std():.4f}")
else:
    print("\nNo evaluation results generated.")

print("\n" + "=" * 80)
print("✓ Training and evaluation completed successfully!")
print("=" * 80)
print(f"Results location: {save_path}")
print(f"  - Model: model.pt")
print(f"  - Evaluation: evaluation_results.csv")
print(f"  - History: history.csv")
print("=" * 80)
