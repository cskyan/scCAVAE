import json
import logging
import os
import pickle
from typing import Optional, Sequence, Union, List, Dict

import torch.nn as nn

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    ObsmField,
)
from anndata import AnnData
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.train._callbacks import SaveBestState
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from ._module import scCAVAEModule
from ._utils import SCAVAE_REGISTRY_KEYS
from ._task import scCAVAETrainingPlan
from ._data import AnnDataSplitter

logger = logging.getLogger(__name__)
logger.propagate = False

class scCAVAE(BaseModelClass):
    

    covars_encoder: dict = None
    pert_encoder: dict = None
    pert_smiles_map: dict = None

    def __init__(
        self,
        adata: AnnData,
        split_key: str = None,
        train_split: Union[str, List[str]] = "train",
        valid_split: Union[str, List[str]] = "test",
        test_split: Union[str, List[str]] = "ood",
        use_rdkit_embeddings: bool = False,
        **hyper_params,
    ):
        super().__init__(adata)

        self.split_key = split_key

        self.drugs = list(self.pert_encoder.keys())
        self.covars = {
            covar: list(self.covars_encoder[covar].keys())
            for covar in self.covars_encoder.keys()
        }

        if use_rdkit_embeddings and self.pert_smiles_map is not None:
            
            drug_embeddings = self.__get_rdkit_embeddings()
            hyper_params['drug_embeddings'] = drug_embeddings

        self.module = scCAVAEModule(
            n_genes=adata.n_vars,
            n_perts=len(self.pert_encoder),
            covars_encoder=self.covars_encoder,
            **hyper_params,
        ).float()

        train_indices, valid_indices, test_indices = None, None, None
        if split_key is not None:
            train_split = (
                train_split if isinstance(train_split, list) else [train_split]
            )
            valid_split = (
                valid_split if isinstance(valid_split, list) else [valid_split]
            )
            test_split = test_split if isinstance(test_split, list) else [test_split]

            train_indices = np.where(adata.obs.loc[:, split_key].isin(train_split))[0]
            valid_indices = np.where(adata.obs.loc[:, split_key].isin(valid_split))[0]
            test_indices = np.where(adata.obs.loc[:, split_key].isin(test_split))[0]

        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices

        self._model_summary_string = f"Single-Cell Causal Autoencoder with Variational Attention and Embedding"

        self.init_params_ = self._get_init_params(locals())

        self.epoch_history = None

    def __get_rdkit_embeddings(
        self,
    ):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError(
                "rdkit is required for use_rdkit_embeddings=True. "
                "Install it with: conda install -c conda-forge rdkit"
            )

        assert self.pert_smiles_map not in [None, []]
        query_drug_names = list(self.pert_encoder.keys())
        query_drug_names.remove('<PAD>')

        smiles_list = [self.pert_smiles_map[drug] for drug in list(query_drug_names)]

        drug_fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            drug_fps.append(np.array(fps))
        
        drug_fps = np.vstack(drug_fps)

        print(drug_fps.shape)

        embeddings = AnnData(X=drug_fps)
        embeddings.obs.index = smiles_list
        embeddings = embeddings[list(smiles_list), :]

        drug_embeddings = nn.Embedding(
            len(self.pert_encoder),
            embeddings.shape[1],
            padding_idx=SCAVAE_REGISTRY_KEYS.PADDING_IDX,
        )
        pad_X = np.zeros(shape=(1, embeddings.n_vars))
        X = np.concatenate((pad_X, embeddings.X), 0)
        drug_embeddings.weight.data.copy_(torch.tensor(X))
        drug_embeddings.weight.requires_grad = False

        return drug_embeddings

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        perturbation_key: str,
        control_group: str,
        dosage_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        smiles_key: Optional[str] = None,
        is_count_data: Optional[bool] = True,
        categorical_covariate_keys: Optional[List[str]] = [],
        deg_uns_key: Optional[str] = None,
        deg_uns_cat_key: Optional[str] = None,
        max_comb_len: int = 2,
        **kwargs,
    ):
        
        SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY = perturbation_key
        SCAVAE_REGISTRY_KEYS.PERTURBATION_DOSAGE_KEY = dosage_key
        SCAVAE_REGISTRY_KEYS.CAT_COV_KEYS = categorical_covariate_keys
        SCAVAE_REGISTRY_KEYS.MAX_COMB_LENGTH = max_comb_len
        SCAVAE_REGISTRY_KEYS.BATCH_KEY = batch_key

        if dosage_key is None:
            print(f'Warning: dosage_key is not set. Setting it to "1.0" for all cells')

            dosage_key = 'CPA_dose_val'

            adata.obs[dosage_key] = adata.obs[perturbation_key].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')])).values

        SCAVAE_REGISTRY_KEYS.PERTURBATION_DOSAGE_KEY = dosage_key

        perturbations = adata.obs[perturbation_key].astype(str).values
        dosages = adata.obs[dosage_key].astype(str).values

        category_key = f"{cls.__name__}_cat"
        keys = categorical_covariate_keys + [perturbation_key]

        if batch_key is not None:
            keys = [batch_key] + keys

        adata.obs[category_key] = adata.obs[keys].apply(lambda x: "_".join(x.astype(str)), axis=1)
        SCAVAE_REGISTRY_KEYS.CATEGORY_KEY = category_key

        if cls.pert_encoder is None:
            
            perts_names_unique = set()
            for d in np.unique(perturbations):
                [perts_names_unique.add(i) for i in d.split("+") if i != control_group]
            perts_names_unique = ["<PAD>", control_group] + sorted(
                list(perts_names_unique)
            )
            SCAVAE_REGISTRY_KEYS.PADDING_IDX = 0

            pert_encoder = {pert: i for i, pert in enumerate(perts_names_unique)}

        else:
            pert_encoder = cls.pert_encoder
            perts_names_unique = list(pert_encoder.keys())

        if smiles_key is not None:
            if cls.pert_smiles_map is None:
                pert_smiles_map = {}
                for pert in perts_names_unique:
                    if pert != "<PAD>":
                        try:
                            pert_smiles_map[pert] = adata.obs.loc[
                                adata.obs[perturbation_key] == pert, smiles_key
                            ].values[0]
                        except:
                            pert_name = adata.obs.loc[
                                adata.obs[perturbation_key].str.contains(pert), perturbation_key
                            ].values[0]

                            smiles = adata.obs.loc[
                                adata.obs[perturbation_key].str.contains(pert), smiles_key
                            ].values[0]

                            pert_smiles_map[pert] = smiles.split('..')[pert_name.split('+').index(pert)]
                cls.pert_smiles_map = pert_smiles_map
            else:
                pert_smiles_map = cls.pert_smiles_map

        pert_map = {}
        for condition in tqdm(perturbations):
            perts_list = np.where(np.isin(perts_names_unique, condition.split("+")))[0]
            pert_map[condition] = list(perts_list) + [
                SCAVAE_REGISTRY_KEYS.PADDING_IDX
                for _ in range(max_comb_len - len(perts_list))
            ]

        dose_map = {}
        for dosage_str in tqdm(dosages):
            dosages_list = [float(i) for i in dosage_str.split("+")]
            dose_map[dosage_str] = list(dosages_list) + [
                0.0 for _ in range(max_comb_len - len(dosages_list))
            ]

        data_perts = np.vstack(
            np.vectorize(lambda x: pert_map[x], otypes=[np.ndarray])(perturbations)
        ).astype(int)
        adata.obsm[SCAVAE_REGISTRY_KEYS.PERTURBATIONS] = data_perts

        data_perts_dosages = np.vstack(
            np.vectorize(lambda x: dose_map[x], otypes=[np.ndarray])(dosages)
        ).astype(float)
        adata.obsm[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = data_perts_dosages

        
        control_key = f"{cls.__name__}_{control_group}"
        SCAVAE_REGISTRY_KEYS.CONTROL_KEY = control_key
        adata.obs[control_key] = (adata.obs[perturbation_key] == control_group).astype(
            int
        )

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(
                registry_key=SCAVAE_REGISTRY_KEYS.X_KEY,
                layer=layer,
                is_count_data=is_count_data,
            ),
            ObsmField(
                SCAVAE_REGISTRY_KEYS.PERTURBATIONS,
                SCAVAE_REGISTRY_KEYS.PERTURBATIONS,
                is_count_data=True,
                correct_data_format=True,
            ),
            ObsmField(
                SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES,
                SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES,
                is_count_data=False,
                correct_data_format=True,
            ),
            CategoricalObsField(
                registry_key=SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY,
                attr_key=perturbation_key,
            ),
        ] + [
            CategoricalObsField(registry_key=covar, attr_key=covar)
            for covar in categorical_covariate_keys
        ]

        anndata_fields.append(
            NumericalObsField(registry_key=control_key, attr_key=control_key)
        )
        anndata_fields.append(
            CategoricalObsField(registry_key=category_key, attr_key=category_key)
        )

        if batch_key is not None:
            anndata_fields.append(
                CategoricalObsField(registry_key=batch_key, attr_key=batch_key)
            )

        if deg_uns_key:
            n_deg_r2 = kwargs.pop("n_deg_r2", 10)

            cov_cond_unique = np.unique(adata.obs[deg_uns_cat_key].astype(str).values)

            cov_cond_map = {}
            cov_cond_map_r2 = {}
            for cov_cond in tqdm(cov_cond_unique):
                if cov_cond in adata.uns[deg_uns_key].keys():
                    mask_hvg = adata.var_names.isin(
                        adata.uns[deg_uns_key][cov_cond]
                    ).astype(int)
                    mask_hvg_r2 = adata.var_names.isin(
                        adata.uns[deg_uns_key][cov_cond][:n_deg_r2]
                    ).astype(int)
                    cov_cond_map[cov_cond] = list(mask_hvg)
                    cov_cond_map_r2[cov_cond] = list(mask_hvg_r2)
                else:
                    no_mask = list(np.ones(shape=(adata.n_vars,)))
                    cov_cond_map[cov_cond] = no_mask
                    cov_cond_map_r2[cov_cond] = no_mask

            mask = np.vstack(
                np.vectorize(lambda x: cov_cond_map[x], otypes=[np.ndarray])(
                    adata.obs[deg_uns_cat_key].astype(str).values
                )
            )
            mask_r2 = np.vstack(
                np.vectorize(lambda x: cov_cond_map_r2[x], otypes=[np.ndarray])(
                    adata.obs[deg_uns_cat_key].astype(str).values
                )
            )

            SCAVAE_REGISTRY_KEYS.DEG_MASK = "deg_mask"
            SCAVAE_REGISTRY_KEYS.DEG_MASK_R2 = "deg_mask_r2"
            adata.obsm[SCAVAE_REGISTRY_KEYS.DEG_MASK] = np.array(mask)
            adata.obsm[SCAVAE_REGISTRY_KEYS.DEG_MASK_R2] = np.array(mask_r2)

            anndata_fields.append(
                ObsmField(
                    SCAVAE_REGISTRY_KEYS.DEG_MASK,
                    SCAVAE_REGISTRY_KEYS.DEG_MASK,
                    is_count_data=True,
                    correct_data_format=True,
                )
            )
            anndata_fields.append(
                ObsmField(
                    SCAVAE_REGISTRY_KEYS.DEG_MASK_R2,
                    SCAVAE_REGISTRY_KEYS.DEG_MASK_R2,
                    is_count_data=True,
                    correct_data_format=True,
                )
            )

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        keys = categorical_covariate_keys.copy()
        if batch_key is not None:
            keys.append(batch_key)

        covars_encoder = {}
        for covar in keys:
            covars_encoder[covar] = {
                c: i
                for i, c in enumerate(
                    adata_manager.registry["field_registries"][covar]["state_registry"][
                        "categorical_mapping"
                    ]
                )
            }

        if cls.covars_encoder is None:
            cls.covars_encoder = covars_encoder

        if cls.pert_encoder is None:
            cls.pert_encoder = pert_encoder

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        save_path: Optional[str] = None,
        check_val_every_n_epoch: int = 10,
        early_stopping_patience: int = 10,
        use_auto_optimization: bool = False,
        **trainer_kwargs,
    ):
        
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        
        
        data_loader_kwargs = {}
        if "num_workers" in trainer_kwargs:
            data_loader_kwargs["num_workers"] = trainer_kwargs.pop("num_workers")
        else:
            data_loader_kwargs["num_workers"] = 4  

        
        if data_loader_kwargs.get("num_workers", 0) > 0:
            data_loader_kwargs.setdefault("persistent_workers", True)
            data_loader_kwargs.setdefault("prefetch_factor", 2)

        manual_splitting = (
            (self.valid_indices is not None)
            and (self.train_indices is not None)
            and (self.test_indices is not None)
        )
        if manual_splitting:
            data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **data_loader_kwargs,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                **data_loader_kwargs,
            )

        perturbation_key = SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY
        pert_adv_encoder = {
            c: i
            for i, c in enumerate(
                self.adata_manager.registry["field_registries"][perturbation_key][
                    "state_registry"
                ]["categorical_mapping"]
            )
        }

        drug_weights = []
        n_adv_perts = len(self.adata.obs[perturbation_key].unique())
        for condition in tqdm(list(pert_adv_encoder.keys())):
            n_positive = len(self.adata[self.adata.obs[perturbation_key] == condition])
            drug_weights.append((self.adata.n_obs / n_positive) - 1.0)

        
        if use_auto_optimization:
            
            from ._task_auto import scCAVAEAutoTrainingPlan
            self.training_plan = scCAVAEAutoTrainingPlan(
                self.module,
                self.covars_encoder,
                **plan_kwargs,
            )
            print("Using scCAVAEAutoTrainingPlan.")
        else:
            
            from ._task import scCAVAETrainingPlan
            self.training_plan = scCAVAETrainingPlan(
                self.module,
                self.covars_encoder,
                n_adv_perts=n_adv_perts,
                **plan_kwargs,
                drug_weights=drug_weights,
            )
            print("Using scCAVAETrainingPlan.")
        trainer_kwargs["early_stopping"] = False
        trainer_kwargs["check_val_every_n_epoch"] = check_val_every_n_epoch

        es_callback = EarlyStopping(
            monitor="scavae_metric",
            patience=early_stopping_patience,
            check_on_train_epoch_end=False,
            verbose=False,
            mode="max",
        )

        if "callbacks" in trainer_kwargs.keys() and isinstance(
            trainer_kwargs.get("callbacks"), list
        ):
            trainer_kwargs["callbacks"] += [es_callback]
        else:
            trainer_kwargs["callbacks"] = [es_callback]

        if save_path is None:
            save_path = "./"

        checkpoint = SaveBestState(
            monitor="scavae_metric", mode="max", period=1, verbose=True
        )
        trainer_kwargs["callbacks"].append(checkpoint)

        self.runner = TrainRunner(
            self,
            training_plan=self.training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping_monitor="scavae_metric",
            early_stopping_mode="max",
            **trainer_kwargs,
        )
        self.runner()

        self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)
        if save_path is not False:
            self.save(save_path, overwrite=True)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 32,
    ):
        

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )

        latent_basal = []
        latent = []
        latent_corrected = []
        for tensors in tqdm(scdl):
            tensors, _ = self.module.mixup_data(tensors, alpha=0.0)
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            latent_basal += [outputs["z_basal"].cpu().numpy()]
            latent += [outputs["z"].cpu().numpy()]
            latent_corrected += [outputs["z_corrected"].cpu().numpy()]

        latent_basal_adata = AnnData(
            X=np.concatenate(latent_basal, axis=0), obs=adata.obs.copy()
        )
        latent_basal_adata.obs_names = adata.obs_names

        latent_corrected_adata = AnnData(
            X=np.concatenate(latent_corrected, axis=0), obs=adata.obs.copy()
        )
        latent_corrected_adata.obs_names = adata.obs_names

        latent_adata = AnnData(X=np.concatenate(latent, axis=0), obs=adata.obs.copy())
        latent_adata.obs_names = adata.obs_names

        latent_outputs = {
            "latent_corrected": latent_corrected_adata,
            "latent_basal": latent_basal_adata,
            "latent_after": latent_adata,
        }

        return latent_outputs

    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 32,
        n_samples: int = 20,
        return_mean: bool = True,
    ):
        
        assert self.module.recon_loss in ["gauss", "nb", "zinb", "mse", "mse_sum", "huber"]
        self.module.eval()

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        xs = []
        for tensors in tqdm(scdl):
            
            expression_results = self.module.get_expression(tensors, n_samples=n_samples)
            
            
            x_pred = expression_results['px'].detach().cpu().numpy()
            xs.append(x_pred)

        if n_samples > 1 and self.module.variational:
            
            x_pred = np.concatenate(xs, axis=1)
        else:
            x_pred = np.concatenate(xs, axis=0)

        if self.module.variational and n_samples > 1 and return_mean:
            x_pred = x_pred.mean(0)
            
        x_pred = np.clip(x_pred, a_min=1e-6, a_max=None) 
        
        adata.obsm[f"{self.__class__.__name__}_pred"] = x_pred

    def custom_predict(
        self,
        covars_to_add: Optional[Sequence[str]] = None,
        basal=False,
        add_batch: bool = True,
        add_pert: bool = True,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 32,
        n_samples: int = 20,
        return_mean: bool = True,
    ) -> AnnData:
        
        if covars_to_add is None:
            covars_to_add = []
        for covar in covars_to_add:
            assert covar in self.module.covars_encoder.keys(
            ), f"covariate {covar} not found in learned covariates"

        if basal:
            latent_key = "z_basal"
        else:
            if add_batch and add_pert:
                latent_key = "z"
            elif add_batch:
                latent_key = "z_no_pert"
            elif add_pert:
                latent_key = "z_corrected"
            else:
                latent_key = "z_no_pert_corrected"

        assert self.module.recon_loss in ["gauss", "nb", "zinb", "mse", "mse_sum", "huber"]
        self.module.eval()

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        xs = []
        zs = []
        z_correcteds = []
        z_no_perts = []
        z_no_pert_correcteds = []
        z_basals = []
        for tensors in tqdm(scdl):
            predictions = self.module.get_expression(
                tensors, n_samples=n_samples, covars_to_add=covars_to_add, latent=latent_key)

            px = predictions['px']
            z = predictions['z']
            z_corrected = predictions['z_corrected']
            z_no_pert = predictions['z_no_pert']
            z_no_pert_corrected = predictions['z_no_pert_corrected']
            z_basal = predictions['z_basal']

            x_pred = (
                px.detach().cpu().numpy()
            )
            xs.append(x_pred)

            z = (
                z.detach().cpu().numpy()
            )
            zs.append(z)

            z_corrected = (
                z_corrected.detach().cpu().numpy()
            )
            z_correcteds.append(z_corrected)

            z_no_pert = (
                z_no_pert.detach().cpu().numpy()
            )
            z_no_perts.append(z_no_pert)

            z_no_pert_corrected = (
                z_no_pert_corrected.detach().cpu().numpy()
            )
            z_no_pert_correcteds.append(z_no_pert_corrected)

            z_basal = (
                z_basal.detach().cpu().numpy()
            )
            z_basals.append(z_basal)

        if n_samples > 1 and self.module.variational:
            
            x_pred = np.concatenate(xs, axis=1)
            z = np.concatenate(zs, axis=1)
            z_corrected = np.concatenate(z_correcteds, axis=1)
            z_no_pert = np.concatenate(z_no_perts, axis=1)
            z_no_pert_corrected = np.concatenate(z_no_pert_correcteds, axis=1)
            z_basal = np.concatenate(z_basals, axis=1)
        else:
            x_pred = np.concatenate(xs, axis=0)
            z = np.concatenate(zs, axis=0)
            z_corrected = np.concatenate(z_correcteds, axis=0)
            z_no_pert = np.concatenate(z_no_perts, axis=0)
            z_no_pert_corrected = np.concatenate(z_no_pert_correcteds, axis=0)
            z_basal = np.concatenate(z_basals, axis=0)

        if self.module.variational and n_samples > 1 and return_mean:
            x_pred = x_pred.mean(0)
            z = z.mean(0)
            z_corrected = z_corrected.mean(0)
            z_no_pert = z_no_pert.mean(0)
            z_no_pert_corrected = z_no_pert_correcteds.mean(0)
            z_basal = z_basal.mean(0)

        latent_x_pred = AnnData(
            X=x_pred, obs=adata.obs.copy()
        )
        latent_x_pred.obs_names = adata.obs_names

        latent_z = AnnData(
            X=z, obs=adata.obs.copy()
        )
        latent_z.obs_names = adata.obs_names

        latent_z_corrected = AnnData(
            X=z_corrected, obs=adata.obs.copy()
        )
        latent_z_corrected.obs_names = adata.obs_names

        latent_z_no_pert = AnnData(
            X=z_no_pert, obs=adata.obs.copy()
        )
        latent_z_no_pert.obs_names = adata.obs_names

        latent_z_no_pert_corrected = AnnData(
            X=z_no_pert_corrected, obs=adata.obs.copy()
        )
        latent_z_no_pert_corrected.obs_names = adata.obs_names

        latent_z_basal = AnnData(
            X=z_basal, obs=adata.obs.copy()
        )
        latent_z_basal.obs_names = adata.obs_names

        latent_outputs = {
            "latent_x_pred": latent_x_pred,
            "latent_z": latent_z,
            "latent_z_corrected": latent_z_corrected,
            "latent_z_no_pert": latent_z_no_pert,
            "latent_z_no_pert_corrected": latent_z_no_pert_corrected,
            "latent_z_basal": latent_z_basal,
        }

        return latent_outputs

    @torch.no_grad()
    def get_pert_embeddings(self, dosage=1.0, pert: Optional[str] = None):
        
        self.module.eval()
        if isinstance(dosage, float):
            if pert is None:
                n_drugs = len(self.pert_encoder)
                treatments = [torch.arange(n_drugs, device=self.device).long().unsqueeze(1)]
                for _ in range(SCAVAE_REGISTRY_KEYS.MAX_COMB_LENGTH - 1):
                    treatments += [torch.zeros(n_drugs, device=self.device).long().unsqueeze(1) + SCAVAE_REGISTRY_KEYS.PADDING_IDX]
                
                treatments = torch.cat(treatments, dim=1) 
                treatments_dosages = [torch.tensor([dosage for _ in range(n_drugs)], device=self.device).float().unsqueeze(1)] 
                for _ in range(SCAVAE_REGISTRY_KEYS.MAX_COMB_LENGTH - 1):
                    treatments_dosages += [torch.zeros(n_drugs, device=self.device).float().unsqueeze(1) + SCAVAE_REGISTRY_KEYS.PADDING_IDX]
                treatments_dosages = torch.cat(treatments_dosages, dim=1) 
            else:
                treatments = [self.pert_encoder[pert]] + [SCAVAE_REGISTRY_KEYS.PADDING_IDX for _ in range(SCAVAE_REGISTRY_KEYS.MAX_COMB_LENGTH - 1)]
                treatments = torch.LongTensor(treatments).to(self.device).unsqueeze(0)

                treatments_dosages = [dosage] + [SCAVAE_REGISTRY_KEYS.PADDING_IDX for _ in range(SCAVAE_REGISTRY_KEYS.MAX_COMB_LENGTH - 1)]
                treatments_dosages = torch.FloatTensor(treatments_dosages).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError

        embeds = self.module.pert_network(treatments, treatments_dosages).detach().cpu().numpy() 
        pert_latent_adata = AnnData(X=embeds)
        pert_latent_adata.obs['pert_name'] = [pert] if pert is not None else self.pert_encoder.keys()

        return pert_latent_adata

    @torch.no_grad()
    def get_covar_embeddings(self, covariate: str, covariate_value: str = None):
        
        
        assert covariate in self.covars_encoder.keys(), f"covariate {covariate} not found in learned covariates"
        self.module.eval()

        if covariate_value is None:
            covar_ids = torch.arange(
                len(self.covars_encoder[covariate]), device=self.device
            ).long().unsqueeze(1)
        else:
            covar_ids = torch.LongTensor(
                [self.covars_encoder[covariate][covariate_value]]
            ).to(self.device).long().unsqueeze(1)
        
        embeddings = self.module.covars_embeddings[covariate](covar_ids).detach().cpu().numpy() 
        
        covar_latent_adata = AnnData(X=embeddings)
        covar_latent_adata.obs[covariate] = [covariate_value] if covariate_value is not None else self.covars_encoder[covariate].keys()

        return covar_latent_adata

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        
        os.makedirs(dir_path, exist_ok=True)

        
        total_dict = {
            "pert_encoder": self.pert_encoder,
            "covars_encoder": self.covars_encoder,
            "pert_smiles_map": self.pert_smiles_map,
        }

        json_dict = json.dumps(total_dict)
        with open(os.path.join(dir_path, "scCAVAE_info.json"), "w") as f:
            f.write(json_dict)

        if isinstance(self.epoch_history, dict):
            self.epoch_history = pd.DataFrame().from_dict(
                self.training_plan.epoch_history
            )
            self.epoch_history.to_csv(
                os.path.join(dir_path, "history.csv"), index=False
            )
        elif isinstance(self.epoch_history, pd.DataFrame):
            self.epoch_history.to_csv(
                os.path.join(dir_path, "history.csv"), index=False
            )

        return super().save(
            dir_path=dir_path,
            overwrite=overwrite,
            save_anndata=save_anndata,
            **anndata_write_kwargs,
        )

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        
        
        info_path = os.path.join(dir_path, "scCAVAE_info.json")
        is_legacy_model = False
        
        if not os.path.exists(info_path):
            old_info_path = os.path.join(dir_path, "CPA_info.json")
            if os.path.exists(old_info_path):
                print("Detected a legacy CPA checkpoint format; enabling compatibility mode.")
                info_path = old_info_path
                is_legacy_model = True
            else:
                raise FileNotFoundError(
                    f"Could not find a model metadata file.\n"
                    f"  New format: {info_path}\n"
                    f"  Legacy format: {old_info_path}"
                )
        
        
        with open(info_path) as f:
            total_dict = json.load(f)

            cls.pert_encoder = total_dict["pert_encoder"]
            cls.covars_encoder = total_dict["covars_encoder"]
            cls.pert_smiles_map = total_dict.get("pert_smiles_map", None)
        
        
        if is_legacy_model:
            print("Remapping legacy class paths from cpa.* to scCAVAE.*")

            from scvi.model.base._utils import _load_saved_files, _initialize_model, _validate_var_names
            from scvi.model._utils import parse_use_gpu_arg

            
            class_name_mapping = {
                'CPAModule': 'scCAVAEModule',
                'CPA': 'scCAVAE',
                'CPATrainingPlan': 'scCAVAETrainingPlan',
                'CPAAutoTrainingPlan': 'scCAVAEAutoTrainingPlan',
                'ComPertAPI': 'scCAVAEAPI',
            }

            class LegacyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    
                    if module.startswith('cpa.'):
                        new_module = module.replace('cpa.', 'scCAVAE.')
                        new_name = class_name_mapping.get(name, name)

                        if new_name != name:
                            print(f"  Remapped class: {module}.{name} -> {new_module}.{new_name}")

                        try:
                            mod = __import__(new_module, fromlist=[new_name])
                            return getattr(mod, new_name)
                        except (ImportError, AttributeError) as e:
                            print(f"  Warning: failed to import {new_module}.{new_name}: {e}")
                            pass

                    return super().find_class(module, name)

            
            original_unpickler = pickle.Unpickler
            pickle.Unpickler = LegacyCompatUnpickler

            try:
                load_adata = adata is None
                _gpu_result = parse_use_gpu_arg(use_gpu)
                device = _gpu_result[-1]  

                attr_dict, var_names, model_state_dict, new_adata = _load_saved_files(
                    dir_path, load_adata, map_location=device,
                )
                adata = new_adata if new_adata is not None else adata
                _validate_var_names(adata, var_names)

                
                registry = attr_dict.pop("registry_")
                if registry.get("model_name") == "CPA":
                    registry["model_name"] = cls.__name__
                    print(f"  Updated model_name: CPA -> {cls.__name__}")

                
                
                old_setup_args = registry.get("setup_args", {})
                valid_keys = {
                    'perturbation_key', 'control_group', 'dosage_key',
                    'batch_key', 'layer', 'smiles_key', 'is_count_data',
                    'categorical_covariate_keys', 'deg_uns_key',
                    'deg_uns_cat_key', 'max_comb_len',
                }
                clean_args = {k: v for k, v in old_setup_args.items() if k in valid_keys}
                print(f"  Recovered valid setup_args keys from the legacy checkpoint: {list(clean_args.keys())}")
                cls.setup_anndata(adata, **clean_args)

                
                deprecated_hyper_params = {'use_optimized_pert_network'}
                init_params = attr_dict.get("init_params_", {})
                hyper_params = init_params.get("kwargs", {}).get("hyper_params", {})
                removed = [k for k in deprecated_hyper_params if k in hyper_params]
                for k in removed:
                    del hyper_params[k]
                if removed:
                    print(f"  Removed deprecated hyper-parameters: {removed}")

                
                model = _initialize_model(cls, adata, attr_dict)
                model.module.on_load(model)

                
                
                
                target_state = model.module.state_dict()
                remapped_state_dict = {}
                skipped_keys = []
                for k, v in model_state_dict.items():
                    nk = k
                    prefix = "pert_network.combination_encoder."
                    if k.startswith(prefix):
                        suffix = k[len(prefix):]
                        
                        if suffix.startswith("combination_enhancer.0."):
                            field = suffix.split(".")[-1]  
                            nk = prefix + "norm2." + field
                        
                        elif suffix.startswith("combination_enhancer.1."):
                            nk = k.replace("combination_enhancer.1.", "combination_enhancer.0.")
                        
                        elif suffix.startswith("combination_enhancer.4."):
                            nk = k.replace("combination_enhancer.4.", "combination_enhancer.2.")
                        
                        elif suffix == "residual_weight":
                            skipped_keys.append(k)
                            continue
                    
                    if nk in target_state and target_state[nk].shape == v.shape:
                        remapped_state_dict[nk] = v
                    elif nk in target_state:
                        skipped_keys.append(f"{k}->{nk} (shape mismatch: {v.shape} vs {target_state[nk].shape})")
                    else:
                        remapped_state_dict[nk] = v

                if skipped_keys:
                    print(f"  Legacy compatibility: skipped {len(skipped_keys)} incompatible keys: {skipped_keys}")

                incompatible = model.module.load_state_dict(remapped_state_dict, strict=False)
                if incompatible.missing_keys:
                    print(f"  Legacy compatibility: initialized {len(incompatible.missing_keys)} missing keys with defaults: {incompatible.missing_keys}")
                if incompatible.unexpected_keys:
                    print(f"  Legacy compatibility: ignored {len(incompatible.unexpected_keys)} unexpected keys: {incompatible.unexpected_keys}")
                model.to_device(device)
                model.module.eval()
                model._validate_anndata(adata)

                print("Successfully loaded the legacy CPA checkpoint.")
            except Exception as e:
                print(f"Model loading failed: {e}")
                raise
            finally:
                pickle.Unpickler = original_unpickler
        else:
            
            from scvi.model.base._utils import _load_saved_files, _initialize_model, _validate_var_names
            from scvi.model._utils import parse_use_gpu_arg

            load_adata = adata is None
            _gpu_result = parse_use_gpu_arg(use_gpu)
            device = _gpu_result[-1]  

            try:
                attr_dict, var_names, model_state_dict, new_adata = _load_saved_files(
                    dir_path, load_adata, map_location=device,
                )
            except (ModuleNotFoundError, AttributeError) as e:
                
                print(f"Detected a pickle compatibility issue; enabling pandas compatibility mode: {e}")

                import types

                class _PandasCompatPickle:
                    
                    def __getattr__(self, name):
                        return getattr(pickle, name)

                    class Unpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            import pandas as _pd
                            if name in ('Int64Index', 'Float64Index', 'UInt64Index'):
                                return _pd.Index
                            if module == 'pandas.core.indexes.numeric':
                                if hasattr(_pd, name):
                                    return getattr(_pd, name)
                                module = 'pandas'
                            if module.startswith('pandas.'):
                                try:
                                    return super().find_class(module, name)
                                except (ModuleNotFoundError, AttributeError):
                                    if hasattr(_pd, name):
                                        return getattr(_pd, name)
                            return super().find_class(module, name)

                compat_pickle = _PandasCompatPickle()
                model_path = os.path.join(dir_path, "model.pt")
                model_data = torch.load(model_path, map_location=device, pickle_module=compat_pickle)
                attr_dict = model_data["attr_dict"]
                var_names = model_data["var_names"]
                model_state_dict = model_data["model_state_dict"]
                new_adata = None

            adata = new_adata if new_adata is not None else adata
            _validate_var_names(adata, var_names)

            registry = attr_dict.pop("registry_")
            setup_args = registry.get("setup_args", {})
            method_name = registry.get("setup_method_name", "setup_anndata")
            getattr(cls, method_name)(adata, source_registry=registry, **setup_args)

            model = _initialize_model(cls, adata, attr_dict)
            model.module.on_load(model)
            model.module.load_state_dict(model_state_dict)
            model.to_device(device)
            model.module.eval()
            model._validate_anndata(adata)

        try:
            model.epoch_history = pd.read_csv(os.path.join(dir_path, "history.csv"))
        except:
            print("WARNING: The history was not found.")

        return model
