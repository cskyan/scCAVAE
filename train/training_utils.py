from __future__ import annotations

import os
import sys
from pathlib import Path

import scanpy as sc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import scCAVAE

PREPARED_DATA_DIR = Path(os.environ.get("SCCAVAE_PREPARED_DATA_DIR", ROOT / "prepared_data"))
RAW_DATA_PATHS = {
    "combosciplex": Path(
        os.environ.get(
            "COMBOSCIPLEX_RAW_PATH",
            "/srv/storage/ssd/ysk/lixin/Dataset/Medicine-Perturb/combo_sciplex_prep_hvg_filtered.h5ad",
        )
    ),
    "kang": Path(
        os.environ.get(
            "KANG_RAW_PATH",
            "/srv/storage/ssd/ysk/lixin/pertpredict/datasets/kang_count.h5ad",
        )
    ),
    "norman": Path(
        os.environ.get(
            "NORMAN_RAW_PATH",
            "/srv/storage/ssd/ysk/lixin/pertpredict/datasets/norman_prepped.h5ad",
        )
    ),
    "sciplex": Path(
        os.environ.get(
            "SCIPLEX_RAW_PATH",
            "/srv/storage/ssd/ysk/lixin/pertpredict/datasets/sciplex_prepped.h5ad",
        )
    ),
}


def load_dataset_for_training(dataset_name: str, use_prepared: bool = True, verbose: bool = True):
    prepared_path = PREPARED_DATA_DIR / f"{dataset_name}_prepared.h5ad"
    if use_prepared and prepared_path.exists():
        if verbose:
            print(f"Loading prepared dataset from {prepared_path}")
        adata = sc.read_h5ad(prepared_path)
        if verbose:
            print(f"  Data shape: {adata.shape}")
        return adata, True
    if verbose:
        print(f"Prepared dataset not found for {dataset_name}; falling back to raw data.")
    return None, False


def setup_sciplex_data(adata=None, verbose: bool = True):
    if adata is None:
        data_path = RAW_DATA_PATHS["sciplex"]
        if verbose:
            print(f"Loading raw Sciplex data from {data_path}")
        adata = sc.read_h5ad(data_path)
    if '(+)-JQ1' in adata.obs['perturbation'].values:
        adata.obs['perturbation'] = adata.obs['perturbation'].replace({'(+)-JQ1': 'JQ1_plus'})
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
    return adata


def setup_kang_data(adata=None, verbose: bool = True):
    if adata is None:
        data_path = RAW_DATA_PATHS["kang"]
        if verbose:
            print(f"Loading raw Kang data from {data_path}")
        adata = sc.read_h5ad(data_path)
    adata.X = adata.layers['counts'].copy()
    adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))
    scCAVAE.setup_anndata(
        adata,
        perturbation_key='condition',
        control_group='ctrl',
        dosage_key='dose',
        categorical_covariate_keys=['cell_type'],
        is_count_data=True,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_drug',
        max_comb_len=1,
    )
    return adata


def setup_norman_data(adata=None, verbose: bool = True):
    if adata is None:
        data_path = RAW_DATA_PATHS["norman"]
        if verbose:
            print(f"Loading raw Norman data from {data_path}")
        adata = sc.read_h5ad(data_path)
    adata.X = adata.layers['counts'].copy()
    scCAVAE.setup_anndata(
        adata,
        perturbation_key='perturbation',
        control_group='ctrl',
        dosage_key='dose',
        categorical_covariate_keys=['cell_type'],
        is_count_data=True,
        deg_uns_key='rank_genes_groups_cov_all',
        deg_uns_cat_key='perturbation',
        max_comb_len=3,
    )
    return adata
