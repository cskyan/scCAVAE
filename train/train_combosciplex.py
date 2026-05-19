from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import scCAVAE
from training_utils import load_dataset_for_training

RESULTS_DIR = ROOT / "train" / "results" / "combosciplex"
RAW_DATA_PATH = Path(
    os.environ.get(
        "COMBOSCIPLEX_RAW_PATH",
        "/srv/storage/ssd/ysk/lixin/Dataset/Medicine-Perturb/combo_sciplex_prep_hvg_filtered.h5ad",
    )
)


def safe_log1p(x):
    if sp.issparse(x):
        return np.log1p(x.toarray())
    return np.log1p(x)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file_obj in self.files:
            file_obj.write(obj)
            file_obj.flush()

    def flush(self):
        for file_obj in self.files:
            file_obj.flush()


def configure_expression_space(adata, recon_loss: str):
    use_log_space = recon_loss in ('mse', 'mse_sum', 'huber', 'gauss')
    if not use_log_space:
        if 'counts' not in adata.layers:
            raise ValueError(
                f"recon_loss={recon_loss!r} requires a counts layer, but only found: {list(adata.layers.keys())}"
            )
        adata.X = adata.layers['counts'].copy()
        x_max = adata.X.max() if not sp.issparse(adata.X) else adata.X.max()
        print(f"  Using raw counts from adata.layers['counts'] (range: 0 - {float(x_max):.0f})")
    else:
        x_max = adata.X.max() if not sp.issparse(adata.X) else adata.X.max()
        if x_max > 20:
            adata.X = np.log1p(adata.X.toarray()) if sp.issparse(adata.X) else np.log1p(adata.X)
            print(f"  Converted expression matrix to log1p space (range: 0 - {adata.X.max():.2f})")
        else:
            print(f"  Expression matrix already appears to be in log1p space (range: 0 - {float(x_max):.2f})")
    return not use_log_space


def evaluate_predictions(adata, save_path: Path, pred_in_count_scale: bool):
    results = defaultdict(list)
    n_top_degs = [10, 20, 50, None]

    for cat in tqdm(adata.obs['cov_drug_dose'].unique(), desc='Evaluating conditions'):
        if 'CHEMBL504' in cat and '+' not in cat.split('_', 1)[1]:
            continue

        parts = cat.split('_', 1)
        if len(parts) < 2:
            continue
        cov = parts[0]
        condition = parts[1]

        cat_obs_names = adata.obs[adata.obs['cov_drug_dose'] == cat].index
        if len(cat_obs_names) == 0:
            continue

        ctrl_candidates = [
            candidate
            for candidate in adata.obs['cov_drug_dose'].unique()
            if candidate.startswith(f'{cov}_CHEMBL504') and '+' not in candidate.split('_', 1)[1]
        ]
        if not ctrl_candidates:
            continue

        ctrl_cat = ctrl_candidates[0]
        ctrl_obs_names = adata.obs[adata.obs['cov_drug_dose'] == ctrl_cat].index
        if len(ctrl_obs_names) == 0:
            continue
        if cat not in adata.uns['rank_genes_groups_cov']:
            continue

        deg_list = adata.uns['rank_genes_groups_cov'][cat]
        cat_adata = adata[cat_obs_names].copy()
        ctrl_adata = adata[ctrl_obs_names].copy()

        x_true = cat_adata.X
        x_pred = cat_adata.obsm['scCAVAE_pred']
        x_ctrl = ctrl_adata.X

        if pred_in_count_scale:
            x_true = safe_log1p(x_true)
            x_pred = safe_log1p(x_pred)
            x_ctrl = safe_log1p(x_ctrl)
        else:
            if sp.issparse(x_true):
                x_true = x_true.toarray()
            if sp.issparse(x_pred):
                x_pred = x_pred.toarray()
            if sp.issparse(x_ctrl):
                x_ctrl = x_ctrl.toarray()

        for n_top_deg in n_top_degs:
            if n_top_deg is None:
                deg_indices = np.arange(adata.n_vars)
                deg_label = 'all'
            else:
                deg_indices = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
                deg_label = n_top_deg

            x_true_deg = x_true[:, deg_indices]
            x_pred_deg = x_pred[:, deg_indices]
            x_ctrl_deg = x_ctrl[:, deg_indices]

            results['condition'].append(condition)
            results['cell_type'].append(cov)
            results['n_top_deg'].append(deg_label)
            results['r2_mean_deg'].append(r2_score(x_true_deg.mean(0), x_pred_deg.mean(0)))
            results['r2_var_deg'].append(r2_score(x_true_deg.var(0), x_pred_deg.var(0)))
            results['r2_mean_lfc_deg'].append(
                r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            )
            results['r2_var_lfc_deg'].append(
                r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))
            )

    df = pd.DataFrame(results)
    results_csv_path = save_path / 'evaluation_results.csv'
    df.to_csv(results_csv_path, index=False)
    print(f"  Saved evaluation results to: {results_csv_path}")

    print("\n" + '=' * 80)
    print('Evaluation Summary')
    print('=' * 80)
    if df.empty:
        print('  WARNING: no evaluation results were generated.')
        return

    for n_deg in [10, 20, 50, 'all']:
        df_subset = df[df['n_top_deg'] == n_deg]
        if len(df_subset) == 0:
            continue
        print(f"\nTop {n_deg} DEGs:")
        print(f"  R2 (mean):     {df_subset['r2_mean_deg'].mean():.4f} +/- {df_subset['r2_mean_deg'].std():.4f}")
        print(f"  R2 (mean LFC): {df_subset['r2_mean_lfc_deg'].mean():.4f} +/- {df_subset['r2_mean_lfc_deg'].std():.4f}")

    ood_conditions = adata.obs[adata.obs['split_1ct_MEC'] == 'ood']['cov_drug_dose'].unique()
    ood_df = df[df.apply(lambda row: f"{row['cell_type']}_{row['condition']}" in ood_conditions, axis=1)]
    if len(ood_df) == 0:
        return

    print("\n" + '=' * 80)
    print('OOD Test Set Performance')
    print('=' * 80)
    for n_deg in [10, 20, 50, 'all']:
        ood_subset = ood_df[ood_df['n_top_deg'] == n_deg]
        if len(ood_subset) == 0:
            continue
        print(f"\nTop {n_deg} DEGs:")
        print(f"  R2 (mean):     {ood_subset['r2_mean_deg'].mean():.4f}")
        print(f"  R2 (mean LFC): {ood_subset['r2_mean_lfc_deg'].mean():.4f}")


def main():
    torch.set_float32_matmul_precision('high')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = RESULTS_DIR / current_time
    save_path.mkdir(parents=True, exist_ok=True)

    log_file = open(save_path / 'training.log', 'w')
    sys.stdout = Tee(sys.stdout, log_file)

    print('=' * 80)
    print('scCAVAE Training - ComboSciPlex Dataset')
    print('=' * 80)
    print(f'Using GPU: {gpu_id}')
    print(f'Results will be saved to: {save_path}')
    print(f'Training log: {save_path / "training.log"}')

    print("\n[1/5] Loading ComboSciPlex dataset...")
    adata, is_prepared = load_dataset_for_training('combosciplex', use_prepared=True)
    if not is_prepared:
        print(f'  Loading raw data from {RAW_DATA_PATH}')
        adata = sc.read_h5ad(RAW_DATA_PATH)
    print(f'  Data shape: {adata.shape}')

    print("\n[2/5] Setting up AnnData...")
    recon_loss = 'nb'
    is_count = configure_expression_space(adata, recon_loss)
    print(f'  is_count_data={is_count}')

    scCAVAE.setup_anndata(
        adata,
        perturbation_key='condition_ID',
        control_group='CHEMBL504',
        dosage_key='log_dose',
        categorical_covariate_keys=['cell_type'],
        is_count_data=is_count,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_drug_dose',
        max_comb_len=2,
    )

    print("\n[3/5] Configuring model...")
    model_params = {
        'n_latent': 256,
        'n_hidden_encoder': 768,
        'n_layers_encoder': 3,
        'n_hidden_decoder': 768,
        'n_layers_decoder': 3,
        'dropout_rate_encoder': 0.1,
        'dropout_rate_decoder': 0.1,
        'use_batch_norm_encoder': True,
        'use_layer_norm_encoder': False,
        'use_batch_norm_decoder': True,
        'use_layer_norm_decoder': False,
        'recon_loss': recon_loss,
        'variational': True,
        'doser_type': 'logsigm',
        'encoding_strategy': 'combination_attention',
        'dose_aware': True,
        'seed': 434,
    }
    trainer_params = {
        'lr': 0.0006,
        'wd': 4e-7,
        'n_epochs_pretrain_ae': 30,
        'n_epochs_mixup_warmup': 3,
        'mixup_alpha': 0.1,
        'step_size_lr': 35,
        'do_clip_grad': True,
        'gradient_clip_value': 1.0,
        'contrastive_loss_type': 'hierarchical_supcon',
        'reg_contrastive': 1.5,
        'supcon_temperature': 0.07,
        'n_epochs_contrastive_warmup': 40,
        'use_adversarial_training': False,
    }

    print("\nModel parameters:")
    for key, value in model_params.items():
        print(f'  {key}: {value}')
    print("\nTrainer parameters:")
    for key, value in trainer_params.items():
        print(f'  {key}: {value}')

    print("\n[4/5] Creating model...")
    model = scCAVAE(
        adata=adata,
        split_key='split_1ct_MEC',
        train_split='train',
        valid_split='valid',
        test_split='ood',
        **model_params,
    )

    print("\n[5/5] Starting training...")
    print('=' * 80)
    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=16384,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=10,
        save_path=str(save_path),
        precision=16,
        use_auto_optimization=True,
    )

    print("\n" + '=' * 80)
    print('Training completed!')
    print(f'Model saved to: {save_path}')
    print('=' * 80)

    print("\n" + '=' * 80)
    print('Evaluating model performance...')
    print('=' * 80)
    model.predict(adata, batch_size=2048)
    recon_loss_type = model.module.recon_loss
    pred_in_count_scale = recon_loss_type in {'nb', 'zinb'}
    print(f"\nDetected reconstruction loss: {recon_loss_type}")
    evaluate_predictions(adata, save_path, pred_in_count_scale)

    print("\n" + '=' * 80)
    print('Training and evaluation completed!')
    print(f'Results location: {save_path}')
    print('=' * 80)


if __name__ == '__main__':
    main()
