import numpy as np
import torch
import torch.nn as nn
from scvi import settings
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, DecoderSCVI
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import pearson_corrcoef, r2_score

from ._metrics import knn_purity
from ._utils import OptimizedPerturbationNetwork, VanillaEncoder, SCAVAE_REGISTRY_KEYS, Decoder, TeLU
import torch.nn.functional as F
from typing import Optional

class scCAVAEModule(BaseModuleClass):
    

    def __init__(self,
                 n_genes: int,
                 n_perts: int,
                 covars_encoder: dict,
                 drug_embeddings: Optional[np.ndarray] = None,
                 n_latent: int = 128,
                 
                 n_latent_basal: Optional[int] = None,  
                 n_latent_pert: Optional[int] = None,   
                 n_latent_covs: Optional[int] = None,   
                 recon_loss: str = "nb",
                 doser_type: str = "logsigm",
                 n_hidden_encoder: int = 256,
                 n_layers_encoder: int = 3,
                 n_hidden_decoder: int = 256,
                 n_layers_decoder: int = 3,
                 n_hidden_doser: int = 128,
                 n_layers_doser: int = 2,
                 use_batch_norm_encoder: bool = True,
                 use_layer_norm_encoder: bool = False,
                 use_batch_norm_decoder: bool = True,
                 use_layer_norm_decoder: bool = False,
                 dropout_rate_encoder: float = 0.0,
                 dropout_rate_decoder: float = 0.0,
                 variational: bool = False,
                 seed: int = 0,
                 use_simple_residual: bool = False,
                 
                 encoding_strategy: str = 'embedding',
                 dose_aware: bool = None,  
                 attention_type: str = 'none',
                 attention_target: str = 'none',
                 attention_reduction: int = 16,
                 
                 latent_fusion_strategy: str = "addition",  
                 ):
        super().__init__()

        recon_loss = recon_loss.lower()
        assert recon_loss in ['gauss', 'zinb', 'nb', 'mse', 'mse_sum', 'huber']
        
        
        assert latent_fusion_strategy in ['addition', 'concatenation', 'attention_fusion'],            f"latent_fusion_strategy must be one of ['addition', 'concatenation', 'attention_fusion'], got {latent_fusion_strategy}"

        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

        self.n_genes = n_genes
        self.n_perts = n_perts
        self.n_latent = n_latent
        self.recon_loss = recon_loss
        self.doser_type = doser_type
        self.variational = variational

        
        self.n_latent_basal = n_latent_basal if n_latent_basal is not None else n_latent
        self.n_latent_pert = n_latent_pert if n_latent_pert is not None else n_latent
        self.n_latent_covs = n_latent_covs if n_latent_covs is not None else n_latent
        
        
        self.latent_fusion_strategy = latent_fusion_strategy
        
        
        if self.latent_fusion_strategy == "addition":
            
            self.n_latent_total = self.n_latent
        elif self.latent_fusion_strategy == "concatenation":
            
            self.n_latent_total = self.n_latent_basal + self.n_latent_pert + self.n_latent_covs
        elif self.latent_fusion_strategy == "attention_fusion":
            
            self.n_latent_total = self.n_latent
        
        if self.latent_fusion_strategy == "addition":
            if not (self.n_latent_basal == self.n_latent_pert == self.n_latent_covs == self.n_latent):
                print(
                    "Warning: addition fusion works best when latent dimensions match. "
                    f"Current values are basal={self.n_latent_basal}, pert={self.n_latent_pert}, covs={self.n_latent_covs}."
                )

        self.covars_encoder = covars_encoder

        self.attention_type = attention_type
        self.attention_target = attention_target
        
        if variational:
            self.encoder = Encoder(
                n_genes,
                self.n_latent_basal,
                var_activation=nn.Softplus(),
                n_hidden=n_hidden_encoder,
                n_layers=n_layers_encoder,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                dropout_rate=dropout_rate_encoder,
                activation_fn=TeLU,
                return_dist=True,
            )
        else:
            self.encoder = VanillaEncoder(
                n_input=n_genes,
                n_output=self.n_latent_basal,
                n_cat_list=[],
                n_hidden=n_hidden_encoder,
                n_layers=n_layers_encoder,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                dropout_rate=dropout_rate_encoder,
                activation_fn=nn.ReLU,
                output_activation='linear',
                use_simple_residual=use_simple_residual,
            )

        if self.recon_loss in ['zinb', 'nb']:
            self.px_r = torch.nn.Parameter(torch.randn(self.n_genes))

            self.decoder = DecoderSCVI(
                n_input=self.n_latent_total,
                n_output=n_genes,
                n_layers=n_layers_decoder,
                n_hidden=n_hidden_decoder,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            )

        elif recon_loss == "gauss":
            self.decoder = Encoder(n_input=self.n_latent_total,
                                   n_output=n_genes,
                                   n_layers=n_layers_decoder,
                                   n_hidden=n_hidden_decoder,
                                   dropout_rate=dropout_rate_decoder,
                                   use_batch_norm=use_batch_norm_decoder,
                                   use_layer_norm=use_layer_norm_decoder,
                                   var_activation=None,
                                   )
        elif recon_loss in ["mse", "mse_sum", "huber"]:
            
            
            if variational:
                
                decoder_use_batch_norm = False
                decoder_use_layer_norm = True
            else:
                
                decoder_use_batch_norm = use_batch_norm_decoder
                decoder_use_layer_norm = use_layer_norm_decoder

            self.decoder = Decoder(
                latent_dim=self.n_latent_total,
                n_genes=n_genes,
                n_layers=n_layers_decoder,
                hidden_dim=n_hidden_decoder,
                dropout=dropout_rate_decoder,
                use_batch_norm=decoder_use_batch_norm,
                use_layer_norm=decoder_use_layer_norm,
                output_activation='gelu'
            )
        else:
            raise Exception('Invalid Loss function for Autoencoder')

        self.encoding_strategy = encoding_strategy

        if dose_aware is None:
            dose_aware = doser_type not in ['none', 'linear']

        self.dose_aware = dose_aware
        self.pert_network = OptimizedPerturbationNetwork(
            n_perts=n_perts,
            n_latent=self.n_latent_pert,
            encoding_strategy=encoding_strategy,
            doser_type=doser_type,
            dose_aware=dose_aware,
            n_hidden=n_hidden_doser,
            n_layers=n_layers_doser,
            dropout_rate=dropout_rate_encoder,
            drug_embeddings=drug_embeddings,
        )

        self.covars_embeddings = nn.ModuleDict(
            {
                key: torch.nn.Embedding(len(unique_covars), self.n_latent_covs)
                for key, unique_covars in self.covars_encoder.items()
            }
        )

        self.metrics = {
            'pearson_r': pearson_corrcoef,
            'r2_score': r2_score
        }

        if self.latent_fusion_strategy == "attention_fusion":
            self.latent_attention = nn.MultiheadAttention(
                embed_dim=max(self.n_latent_basal, self.n_latent_pert, self.n_latent_covs),  
                num_heads=4,  
                dropout=0.1,
                batch_first=True
            )
            
            proj_dim = max(self.n_latent_basal, self.n_latent_pert, self.n_latent_covs)
            self.basal_proj = nn.Linear(self.n_latent_basal, proj_dim) if self.n_latent_basal != proj_dim else nn.Identity()
            self.pert_proj = nn.Linear(self.n_latent_pert, proj_dim) if self.n_latent_pert != proj_dim else nn.Identity()
            self.covs_proj = nn.Linear(self.n_latent_covs, proj_dim) if self.n_latent_covs != proj_dim else nn.Identity()
            
            
            self.fusion_output = nn.Sequential(
                nn.Linear(proj_dim * 3, self.n_latent_total),  
                nn.ReLU(),
                nn.Dropout(0.1)
            )

    def mixup_data(self, tensors, alpha: float = 0.0, opt=False):
        
        alpha = max(0.0, alpha)

        if alpha == 0.0:
            
            tensors[SCAVAE_REGISTRY_KEYS.X_KEY + '_true'] = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]
            tensors[SCAVAE_REGISTRY_KEYS.X_KEY + '_mixup'] = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]
            tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY + '_mixup'] = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY]
            tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS + '_mixup'] = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS]
            tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES + '_mixup'] = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES]
            for covar in self.covars_encoder.keys():
                tensors[covar + '_mixup'] = tensors[covar]
            return tensors, 1.0

        mixup_lambda = np.random.beta(alpha, alpha)

        x = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]
        y_perturbations = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY]
        perturbations = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS]
        perturbations_dosages = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES]

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = mixup_lambda * x + (1. - mixup_lambda) * x[index, :]

        tensors[SCAVAE_REGISTRY_KEYS.X_KEY] = mixed_x
        tensors[SCAVAE_REGISTRY_KEYS.X_KEY + '_true'] = x
        tensors[SCAVAE_REGISTRY_KEYS.X_KEY + '_mixup'] = x[index]
        tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY + '_mixup'] = y_perturbations[index]
        tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS + '_mixup'] = perturbations[index]
        tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES + '_mixup'] = perturbations_dosages[index]

        for covar, encoder in self.covars_encoder.items():
            tensors[covar + '_mixup'] = tensors[covar][index]

        return tensors, mixup_lambda

    def _get_inference_input(self, tensors):
        x = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]
        
        perts = {
            'true': tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS],
            'mixup': tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS + '_mixup']
        }
        perts_doses = {
            'true': tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES],
            'mixup': tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS_DOSAGES + '_mixup'],
        }

        covars_dict = dict()
        for covar, unique_covars in self.covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  
            encoded_covars_mixup = tensors[covar + '_mixup'].view(-1, )  
            covars_dict[covar] = encoded_covars
            covars_dict[covar + '_mixup'] = encoded_covars_mixup

        return dict(
            x=x,
            perts=perts,
            perts_doses=perts_doses,
            covars_dict=covars_dict,
        )

    @auto_move_data
    def inference(
            self,
            x,  
            perts,
            perts_doses,
            covars_dict,
            mixup_lambda: float = 1.0,
            n_samples: int = 1,
            covars_to_add: Optional[list] = None,
            compute_all_latents: bool = True,  
    ):
        batch_size = x.shape[0]
        
        
        if self.recon_loss in ['nb', 'zinb']:
            
            x_ = torch.log1p(x)
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:
            x_ = x
            library = None, None

        
        if self.variational:
            qz, z_basal = self.encoder(x_)
            
        else:
            qz, z_basal = None, self.encoder(x_)
            
        if self.variational and n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)
            
            if self.recon_loss in ['nb', 'zinb']:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            elif self.recon_loss in ['mse', 'mse_sum', 'huber']:
                
                
                pass
                

        
        z_pert_true = self.pert_network(perts['true'], perts_doses['true'])
        if mixup_lambda < 1.0:
            z_pert_mixup = self.pert_network(perts['mixup'], perts_doses['mixup'])
            z_pert = mixup_lambda * z_pert_true + (1. - mixup_lambda) * z_pert_mixup
        else:
            z_pert = z_pert_true
            

        
        if self.variational and n_samples > 1:
            
            
            z_pert = z_pert.unsqueeze(0).expand(n_samples, -1, -1)

        
        
        if self.variational and n_samples > 1:
            z_covs = torch.zeros(n_samples, batch_size, self.n_latent_covs, device=z_basal.device)
            if compute_all_latents:
                z_covs_wo_batch = torch.zeros(n_samples, batch_size, self.n_latent_covs, device=z_basal.device)
        else:
            z_covs = torch.zeros(batch_size, self.n_latent_covs, device=z_basal.device)
            if compute_all_latents:
                z_covs_wo_batch = torch.zeros(batch_size, self.n_latent_covs, device=z_basal.device)

        batch_key = SCAVAE_REGISTRY_KEYS.BATCH_KEY
        
        if covars_to_add is None:
            covars_to_add = list(self.covars_encoder.keys())
            
        for covar, encoder in self.covars_encoder.items():
            if covar in covars_to_add:
                z_cov = self.covars_embeddings[covar](covars_dict[covar].long())
                if len(encoder) > 1:
                    z_cov_mixup = self.covars_embeddings[covar](covars_dict[covar + '_mixup'].long())
                    z_cov = mixup_lambda * z_cov + (1. - mixup_lambda) * z_cov_mixup
                
                if self.variational and n_samples > 1:
                    z_cov = z_cov.view(batch_size, self.n_latent_covs)  
                    z_cov = z_cov.unsqueeze(0).expand(n_samples, -1, -1)  
                else:
                    z_cov = z_cov.view(batch_size, self.n_latent_covs)  
                z_covs += z_cov

                if compute_all_latents and covar != batch_key:
                    z_covs_wo_batch += z_cov

        
        if self.latent_fusion_strategy == "addition":
            
            z = z_basal + z_pert + z_covs
            if compute_all_latents:
                z_corrected = z_basal + z_pert + z_covs_wo_batch
                z_no_pert = z_basal + z_covs
                z_no_pert_corrected = z_basal + z_covs_wo_batch

        elif self.latent_fusion_strategy == "concatenation":
            
            
            
            z = torch.cat([z_basal, z_pert, z_covs], dim=-1)
            if compute_all_latents:
                z_corrected = torch.cat([z_basal, z_pert, z_covs_wo_batch], dim=-1)
                z_no_pert = torch.cat([z_basal, z_covs], dim=-1)
                z_no_pert_corrected = torch.cat([z_basal, z_covs_wo_batch], dim=-1)

        elif self.latent_fusion_strategy == "attention_fusion":
            
            z = self._attention_fusion(z_basal, z_pert, z_covs)
            if compute_all_latents:
                z_corrected = self._attention_fusion(z_basal, z_pert, z_covs_wo_batch)
                z_no_pert = self._attention_fusion(z_basal, torch.zeros_like(z_pert), z_covs)
                z_no_pert_corrected = self._attention_fusion(z_basal, torch.zeros_like(z_pert), z_covs_wo_batch)

        result = dict(
            z=z,
            z_basal=z_basal,
            z_covs=z_covs,
            z_pert=z_pert,
            library=library,
            qz=qz,
            mixup_lambda=mixup_lambda,
        )
        if compute_all_latents:
            result['z_corrected'] = z_corrected
            result['z_no_pert'] = z_no_pert
            result['z_no_pert_corrected'] = z_no_pert_corrected
        return result

    def _attention_fusion(self, z_basal, z_pert, z_covs):
        
        
        original_shape = z_basal.shape[:-1]  
        
        
        z_basal_2d = z_basal.reshape(-1, z_basal.shape[-1])
        z_pert_2d = z_pert.reshape(-1, z_pert.shape[-1])
        z_covs_2d = z_covs.reshape(-1, z_covs.shape[-1])
        
        
        z_basal_proj = self.basal_proj(z_basal_2d)  
        z_pert_proj = self.pert_proj(z_pert_2d)     
        z_covs_proj = self.covs_proj(z_covs_2d)     
        
        
        embeddings = torch.stack([z_basal_proj, z_pert_proj, z_covs_proj], dim=1)
        
        
        attn_output, _ = self.latent_attention(embeddings, embeddings, embeddings)
        
        
        
        attn_output_flat = attn_output.reshape(attn_output.shape[0], -1)
        fused = self.fusion_output(attn_output_flat)
        
        
        output_shape = original_shape + (self.n_latent_total,)
        return fused.reshape(output_shape)

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        if 'latent' in kwargs.keys():
            if kwargs['latent'] in inference_outputs.keys(): 
                z = inference_outputs[kwargs['latent']]
            else:
                raise Exception('Invalid latent space')
        else:
            z = inference_outputs["z"]
        library = inference_outputs['library']

        return dict(
            z=z,
            library=library,
        )

    @auto_move_data
    def generative(
            self,
            z,
            library=None,
    ):
        if self.recon_loss == 'nb':
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)
            px = NegativeBinomial(mu=px_rate, theta=px_r)

        elif self.recon_loss == 'zinb':
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)

            px = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        elif self.recon_loss in ["mse", "mse_sum", "huber"]:  
            predicted_expression = self.decoder(z) 
            return dict(
                x_recon=predicted_expression,  
                px=predicted_expression,       
                pz=None,  
            )
        else:
            px_mean, px_var, x_pred = self.decoder(z)

            px = Normal(loc=px_mean, scale=px_var.sqrt())

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(px=px, pz=pz)

    def loss(self, tensors, inference_outputs, generative_outputs,
             contrastive_loss_type: str = "none",
             supcon_temperature: float = 0.1,
             n_hard_negatives: int = 5,
             n_hard_positives: int = 3,  
             use_cell_type_contrast: bool = True, 
             use_global_hierarchical: bool = False,  
             infonce_temperature: float = 0.15,
             multilevel_contrastive: bool = True,
             pert_contrastive_weight: float = 1.0,
             full_contrastive_weight: float = 0.0,
             false_negative_threshold: float = 0.85,  
            ):
        
        x = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]
        batch_size = x.shape[0]
        
        if self.recon_loss == "mse":
            
            predicted_expression = generative_outputs['x_recon']  
            recon_loss = F.mse_loss(predicted_expression, x, reduction="none")  
            recon_loss = recon_loss.sum(dim=1).mean()  
        elif self.recon_loss == "huber":
            
            predicted_expression = generative_outputs['x_recon']  
            recon_loss = F.huber_loss(predicted_expression, x, reduction="none", delta=1.0)  
            recon_loss = recon_loss.sum(dim=1).mean()  
            
        elif self.recon_loss == "mse_sum":
            
            predicted_expression = generative_outputs['x_recon']
            target_x = torch.log1p(x)
            
            recon_loss = F.mse_loss(predicted_expression, target_x, reduction="sum") / batch_size
        elif self.recon_loss == "gauss":
            
            px = generative_outputs['px']
            px_mean = px.loc
            px_var = px.scale ** 2
            recon_loss = F.gaussian_nll_loss(px_mean, x, px_var, reduction="none").sum(dim=-1).mean()
        else:
            
            px = generative_outputs['px']
            recon_loss = -px.log_prob(x).sum(dim=-1).mean()
            
        
        if self.variational:
            qz = inference_outputs["qz"]
            
            z_basal = inference_outputs["z_basal"]
            pz_basal = Normal(torch.zeros_like(z_basal), torch.ones_like(z_basal))
            kl_divergence_z = kl(qz, pz_basal).sum(dim=1)
            kl_loss = kl_divergence_z.mean()
        else:
            kl_loss = torch.zeros_like(recon_loss)

        contrastive_loss_type = contrastive_loss_type.lower()
        contrastive_loss_value = x.new_tensor(0.0)
        if contrastive_loss_type == "none":
            return recon_loss, kl_loss, contrastive_loss_value

        if contrastive_loss_type == "supcon":
            contrastive_loss_value = self.compute_standard_supcon_loss(
                inference_outputs=inference_outputs,
                tensors=tensors,
                temperature=supcon_temperature,
                use_cell_type_contrast=use_cell_type_contrast,
            )
        elif contrastive_loss_type == "hierarchical_supcon":
            contrastive_loss_value = self.compute_hierarchical_supcon_loss(
                inference_outputs=inference_outputs,
                tensors=tensors,
                temperature=supcon_temperature,
            )
        else:
            raise ValueError(
                "Unsupported contrastive_loss_type. Supported values are 'none', 'supcon', and 'hierarchical_supcon'."
            )

        return recon_loss, kl_loss, contrastive_loss_value

    def compute_standard_supcon_loss(self, inference_outputs, tensors, temperature=0.1, contrast_mode='all', use_cell_type_contrast=True):
        
        device = inference_outputs['z'].device

        
        features = inference_outputs['z']

        
        pert_labels = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY]
         
        cell_type_labels = None
        if 'cell_type' in tensors:
            cell_type_labels = tensors['cell_type']
        elif hasattr(self, 'covars_encoder') and 'cell_type' in self.covars_encoder:
            
            for key in tensors.keys():
                if 'cell_type' in key.lower():
                    cell_type_labels = tensors[key]
                    break
        
        
        if cell_type_labels is not None:
            
            if hasattr(self, 'covars_encoder') and 'cell_type' in self.covars_encoder:
                
                max_cell_types = len(self.covars_encoder['cell_type'])
            else:
                
                max_cell_types = cell_type_labels.max().item() + 1
            
            
            combined_labels = pert_labels * max_cell_types + cell_type_labels
        else:
            
            combined_labels = pert_labels
            print("Warning: cell-type labels were not found, so the contrastive loss will use perturbation labels only.")
        batch_size = features.shape[0]

        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        
        features = torch.nn.functional.normalize(features, p=2, dim=1)

       
        
        labels = combined_labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("The number of labels must match the batch size.")

        
        
        mask = torch.eq(labels, labels.T).float().to(device)

        
        if contrast_mode not in ['one', 'all']:
            raise ValueError("contrast_mode must be either 'one' or 'all'.")

        
        anchor_feature = features
        contrast_feature = features

        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature
        )

        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        
        mask = mask * logits_mask

        
        exp_logits = torch.exp(logits) * logits_mask

        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        
        loss = -mean_log_prob_pos.mean()

        
        if torch.isnan(loss):
            return torch.tensor(0.0, device=device)

        return loss
    

    def compute_hierarchical_supcon_loss(self, inference_outputs, tensors, 
                                           temperature=0.1,
                                           related_weight_alpha=0.1,
                                           related_weight_beta=1.0,
                                           pert_weight=1.0, 
                                           covariate_weight=2.0):
        
        device = inference_outputs['z'].device
        features = F.normalize(inference_outputs['z'], p=2, dim=1)
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        
        perts_ids = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATIONS].long()
        categorical_covariate_keys = SCAVAE_REGISTRY_KEYS.CAT_COV_KEYS
        
        
        perts_a, perts_b = perts_ids.unsqueeze(1), perts_ids.unsqueeze(0)
        valid_a, valid_b = (perts_a != 0).float(), (perts_b != 0).float()
        intersection = ((perts_a == perts_b) & (perts_a != 0)).sum(dim=2).float()
        
        
        max_pert_per_sample = valid_a.sum(dim=2).max()
        if max_pert_per_sample > 1:
            max_pert_id = perts_ids.max().item()
            if max_pert_id > 0:
                pert_presence = (perts_ids.unsqueeze(-1) == torch.arange(1, max_pert_id + 1, device=device).view(1, 1, -1)).any(dim=1).float()
                true_intersection = torch.matmul(pert_presence, pert_presence.T)
                intersection = torch.where((intersection == 0) & (true_intersection > 0), true_intersection, intersection)
        
        union = valid_a.sum(dim=2) + valid_b.sum(dim=2) - intersection
        jaccard_sim = torch.where(union > 0, intersection / union, 
                                torch.where((valid_a.sum(dim=2) == 0) & (valid_b.sum(dim=2) == 0), 
                                        torch.ones_like(intersection), torch.zeros_like(intersection)))
        
        
        covariate_sim = torch.zeros(batch_size, batch_size, device=device)
        
        if categorical_covariate_keys:
            available_covars = []
            available_weights = []
            
            
            for i, covar_key in enumerate(categorical_covariate_keys):
                if covar_key in tensors:
                    available_covars.append(covar_key)
                    available_weights.append(2.0 if i == 0 else 1.0)
            
            
            if available_covars:
                for covar_key, weight in zip(available_covars, available_weights):
                    covar_tensor = tensors[covar_key]
                    covar_similarity = torch.eq(covar_tensor.view(-1, 1), covar_tensor.view(1, -1)).float()
                    covariate_sim += weight * covar_similarity
                
                
                total_weight = sum(available_weights)
                covariate_sim = covariate_sim / total_weight
            else:
                covariate_sim = torch.ones(batch_size, batch_size, device=device)
        else:
            covariate_sim = torch.ones(batch_size, batch_size, device=device)
        
        
        with torch.cuda.amp.autocast():
            sim_matrix = torch.matmul(features, features.T) / temperature
            
            logits_max = torch.max(sim_matrix, dim=1, keepdim=True)[0]
            logits = sim_matrix - logits_max.detach()
        
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        
        pert_pos = (jaccard_sim == 1.0) & (covariate_sim == 1.0)  
        pert_pos.masked_fill_(identity_mask, 0)
        pert_neg = (jaccard_sim == 0.0) & (covariate_sim == 1.0)  
        pert_neg.masked_fill_(identity_mask, 0)
        pert_related = (jaccard_sim > 0) & (jaccard_sim < 1.0) & (covariate_sim == 1.0)
        
        
        exp_logits = torch.exp(logits)
        dynamic_alpha = torch.clamp(related_weight_alpha * torch.exp(-related_weight_beta * jaccard_sim), min=1e-6, max=0.5)
        
        pert_denom = (
            (exp_logits * pert_pos.float()).sum(1, keepdim=True) +
            (exp_logits * pert_neg.float()).sum(1, keepdim=True) +
            (dynamic_alpha * exp_logits * pert_related.float()).sum(1, keepdim=True)
        )
        
        
        pert_log_prob = logits - torch.log(pert_denom + 1e-8)
        pert_pos_counts = pert_pos.sum(1)
        pert_valid = pert_pos_counts > 0
        
        pert_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pert_valid.sum() > 0:
            pert_loss = -(pert_log_prob * pert_pos.float()).sum(1) / (pert_pos_counts.float() + 1e-8)
            pert_loss = pert_loss[pert_valid].mean()
        
        
        covariate_pos = (jaccard_sim == 1.0) & (covariate_sim == 1.0)  
        covariate_pos.masked_fill_(identity_mask, 0)
        
        
        covariate_neg_partial = (jaccard_sim == 1.0) & (covariate_sim < 1.0) & (covariate_sim > 0)  
        covariate_neg_complete = (jaccard_sim == 1.0) & (covariate_sim == 0.0)  
        
        
        soft_weight = 1.0 - covariate_sim
        
        covariate_denom = (
            (exp_logits * covariate_pos.float()).sum(1, keepdim=True) +
            (exp_logits * covariate_neg_complete.float()).sum(1, keepdim=True) +
            (soft_weight * exp_logits * covariate_neg_partial.float()).sum(1, keepdim=True)
        )
        
        covariate_log_prob = logits - torch.log(covariate_denom + 1e-8)
        covariate_pos_counts = covariate_pos.sum(1)
        covariate_valid = covariate_pos_counts > 0
        
        covariate_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if covariate_valid.sum() > 0:
            covariate_loss = -(covariate_log_prob * covariate_pos.float()).sum(1) / (covariate_pos_counts.float() + 1e-8)
            covariate_loss = covariate_loss[covariate_valid].mean()
        
        
        total_weight = pert_weight + covariate_weight
        total_loss = (pert_weight * pert_loss + covariate_weight * covariate_loss) / total_weight
        
        return torch.clamp(total_loss, min=1e-8)

    def r2_metric(self, tensors, inference_outputs, generative_outputs, mode: str = 'lfc'):
        
        mode = mode.lower()
        assert mode in ['direct']

        x = tensors[SCAVAE_REGISTRY_KEYS.X_KEY]  
        indices = tensors[SCAVAE_REGISTRY_KEYS.CATEGORY_KEY].view(-1,)

        unique_indices = indices.unique()

        r2_mean = 0.0
        r2_var = 0.0
        
        
        px = generative_outputs['px']

        for ind in unique_indices:
            i_mask = indices == ind

            x_i = x[i_mask, :]
            
            if self.recon_loss in ['mse', 'mse_sum', 'huber']:
                
                x_pred = generative_outputs['x_recon'][i_mask, :] 
                

                
                x_pred = torch.nan_to_num(x_pred, nan=0.0, posinf=1e3, neginf=-1e3)
                x_i = torch.nan_to_num(x_i, nan=0.0, posinf=1e3, neginf=-1e3)

                if SCAVAE_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                    deg_mask = tensors[f'{SCAVAE_REGISTRY_KEYS.DEG_MASK_R2}'][i_mask, :]

                    x_i *= deg_mask
                    x_pred *= deg_mask

                r2_mean += torch.nan_to_num(self.metrics['r2_score'](x_pred.mean(0), x_i.mean(0)),nan=0.0).item()
                r2_var += 0.0 
            
            elif self.recon_loss == 'gauss' and px is not None:
                x_pred_mean = px.loc[i_mask, :]
                x_pred_var = px.scale[i_mask, :] ** 2

                if SCAVAE_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                    deg_mask = tensors[f'{SCAVAE_REGISTRY_KEYS.DEG_MASK_R2}'][i_mask, :]

                    x_i *= deg_mask
                    x_pred_mean *= deg_mask
                    x_pred_var *= deg_mask

                x_pred_mean = torch.nan_to_num(x_pred_mean, nan=0, posinf=1e3, neginf=-1e3)
                x_pred_var = torch.nan_to_num(x_pred_var, nan=0, posinf=1e3, neginf=-1e3)

                r2_mean += torch.nan_to_num(self.metrics['r2_score'](x_pred_mean.mean(0), x_i.mean(0)),
                                        nan=0.0).item()
                r2_var += torch.nan_to_num(self.metrics['r2_score'](x_pred_var.mean(0), x_i.var(0)),
                                        nan=0.0).item()

            elif self.recon_loss in ['nb', 'zinb'] and px is not None:
                x_i = torch.log(1 + x_i)
                x_pred = px.mu[i_mask, :]
                x_pred = torch.log(1 + x_pred)

                x_pred = torch.nan_to_num(x_pred, nan=0, posinf=1e3, neginf=-1e3)

                if SCAVAE_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                    deg_mask = tensors[f'{SCAVAE_REGISTRY_KEYS.DEG_MASK_R2}'][i_mask, :]

                    x_i *= deg_mask
                    x_pred *= deg_mask

                r2_mean += torch.nan_to_num(self.metrics['r2_score'](x_pred.mean(0), x_i.mean(0)),
                                        nan=0.0).item()
                r2_var += torch.nan_to_num(self.metrics['r2_score'](x_pred.var(0), x_i.var(0)),
                                        nan=0.0).item()

        n_unique_indices = len(unique_indices)
        return r2_mean / n_unique_indices, r2_var / n_unique_indices

    def disentanglement(self, tensors, inference_outputs, generative_outputs, linear=True):
        z_basal = inference_outputs['z_basal'].detach().cpu().numpy()
        z = inference_outputs['z'].detach().cpu().numpy()

        perturbations = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY].view(-1, )
        perturbations_names = perturbations.detach().cpu().numpy()

        knn_basal = knn_purity(z_basal, perturbations_names.ravel(),
                               n_neighbors=min(perturbations_names.shape[0] - 1, 30))
        knn_after = knn_purity(z, perturbations_names.ravel(),
                               n_neighbors=min(perturbations_names.shape[0] - 1, 30))

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                target_covars = tensors[f'{covar}'].detach().cpu().numpy()

                knn_basal += knn_purity(z_basal, target_covars.ravel(),
                                        n_neighbors=min(target_covars.shape[0] - 1, 30))

                knn_after += knn_purity(z, target_covars.ravel(),
                                        n_neighbors=min(target_covars.shape[0] - 1, 30))

        return knn_basal, knn_after

    def get_expression(self, tensors, n_samples=1, covars_to_add=None, latent='z'):
        
        tensors, _ = self.mixup_data(tensors, alpha=0.0)

        inference_outputs, generative_outputs = self.forward(
            tensors,
            inference_kwargs={'n_samples': n_samples, 'covars_to_add': covars_to_add},
            get_generative_input_kwargs={'latent': latent},
            compute_loss=False,
        )

        z = inference_outputs['z']
        z_corrected = inference_outputs['z_corrected']
        z_no_pert = inference_outputs['z_no_pert']
        z_no_pert_corrected = inference_outputs['z_no_pert_corrected']
        z_basal = inference_outputs['z_basal']
        
        if self.recon_loss in ['mse', 'mse_sum', 'huber']:
            
            reconstruction = generative_outputs['x_recon']

            return dict(
                px=reconstruction,           
                x_recon=reconstruction,      
                z=z,
                z_corrected=z_corrected,
                z_no_pert=z_no_pert,
                z_no_pert_corrected=z_no_pert_corrected,
                z_basal=z_basal,
            )
        
        
        px = generative_outputs['px']
        if self.recon_loss == 'gauss':
            reconstruction = px.loc
        else:
            reconstruction = px.mu

        return dict(
            px=reconstruction,
            z=z,
            z_corrected=z_corrected,
            z_no_pert=z_no_pert,
            z_no_pert_corrected=z_no_pert_corrected,
            z_basal=z_basal,
        )

    def get_pert_embeddings(self, tensors, **inference_kwargs):
        inputs = self._get_inference_input(tensors)
        drugs = inputs['perts']
        doses = inputs['perts_doses']

        return self.pert_network(drugs, doses)
