from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial

from scvi.nn import FCLayers
from torch.distributions import Normal
from typing import Optional

class _REGISTRY_KEYS:
    X_KEY: str = "X"
    BATCH_KEY: str = None
    CATEGORY_KEY: str = "scavae_category"
    PERTURBATION_KEY: str = None
    PERTURBATION_DOSAGE_KEY: str = None
    PERTURBATIONS: str = "perts"
    PERTURBATIONS_DOSAGES: str = "perts_doses"
    SIZE_FACTOR_KEY: str = "size_factor"
    CAT_COV_KEYS: List[str] = []
    MAX_COMB_LENGTH: int = 2
    CONTROL_KEY: str = None
    DEG_MASK: str = None
    DEG_MASK_R2: str = None
    PADDING_IDX: int = 0
    
class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

SCAVAE_REGISTRY_KEYS = _REGISTRY_KEYS()

class VanillaEncoder(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
            n_cat_list,
            use_layer_norm=True,
            use_batch_norm=False,
            output_activation: str = 'linear',
            dropout_rate: float = 0.1,
            activation_fn=nn.ReLU,
            use_simple_residual=False,
    ):
        super().__init__()
        self.n_output = n_output
        self.output_activation = output_activation

        self.network = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
        )
        self.z = nn.Linear(n_hidden, n_output)

    def forward(self, inputs, *cat_list):
        
        hidden = self.network(inputs, *cat_list)

        if self.output_activation == 'linear':
            z = self.z(hidden)
        elif self.output_activation == 'relu':
            z = F.relu(self.z(hidden))
        else:
            raise ValueError(f'Unknown output activation: {self.output_activation}')
        return z

class GeneralizedSigmoid(nn.Module):
    

    def __init__(self, n_drugs, non_linearity='sigmoid'):
        
        super(GeneralizedSigmoid, self).__init__()
        self.non_linearity = non_linearity
        self.n_drugs = n_drugs

        self.beta = torch.nn.Parameter(
            torch.ones(1, n_drugs),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, n_drugs),
            requires_grad=True
        )

        self.vmap = None

    def forward(self, x, y):
        
        y = y.long()
        if self.non_linearity == 'logsigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (x * beta + bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.non_linearity == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x

class CombinationAttentionEncoder(nn.Module):
    def __init__(self,
                 n_perts,
                 n_latent,
                 num_heads=8,
                 dropout_rate=0.1,
                 use_layer_norm=True,
                 drug_embeddings=None):
        super().__init__()

        self.n_perts = n_perts
        self.n_latent = n_latent
        self.num_heads = num_heads

        
        if drug_embeddings is not None:
            self.pert_embedding = drug_embeddings
            self.pert_transformation = nn.Linear(drug_embeddings.embedding_dim, n_latent)
            self.use_rdkit = True
        else:
            self.use_rdkit = False
            self.pert_embedding = nn.Embedding(n_perts, n_latent,
                                             padding_idx=SCAVAE_REGISTRY_KEYS.PADDING_IDX)

        
        self.combination_attention = nn.MultiheadAttention(
            embed_dim=n_latent,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        
        self.combination_enhancer = nn.Sequential(
            
            nn.Linear(n_latent, n_latent),
            
            
            TeLU(),
            
            nn.Linear(n_latent, n_latent)
        )
        self.norm1 = nn.LayerNorm(n_latent) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(n_latent) if use_layer_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        
        

    def forward(self, perts):
        
        perts = perts.long()
        bs, max_comb_len = perts.shape

        
        pert_embeddings = self.pert_embedding(perts)  

        if self.use_rdkit:
            pert_embeddings = self.pert_transformation(
                pert_embeddings.view(bs * max_comb_len, -1)
            ).view(bs, max_comb_len, -1)

        
        padding_mask = (perts == SCAVAE_REGISTRY_KEYS.PADDING_IDX)  

        
        normed_embeddings = self.norm1(pert_embeddings)
        attn_output, attn_weights = self.combination_attention(
            normed_embeddings, normed_embeddings, normed_embeddings,
            key_padding_mask=padding_mask
        )  
        x = pert_embeddings + self.dropout1(attn_output)
        
        normed_x = self.norm2(x)
        enhanced_output = self.combination_enhancer(normed_x)

        
        final_output = x + self.dropout2(enhanced_output)
        
        

        return final_output, attn_weights

class OptimizedPerturbationNetwork(nn.Module):
    

    def __init__(self,
                 n_perts,
                 n_latent,
                 encoding_strategy='embedding',
                 doser_type='none',
                 dose_aware=False,
                 n_hidden=None,
                 n_layers=None,
                 dropout_rate=0.0,
                 drug_embeddings=None):
        super().__init__()

        self.n_perts = n_perts
        self.n_latent = n_latent
        self.encoding_strategy = encoding_strategy
        self.doser_type = doser_type
        self.dose_aware = dose_aware

        
        self._init_perturbation_encoders(drug_embeddings)
        
        
        self._init_dose_modeling(n_hidden, n_layers, dropout_rate)

    def _init_perturbation_encoders(self, drug_embeddings):
        
        if self.encoding_strategy == 'onehot':
            
            self.pert_encoder = nn.Linear(self.n_perts, self.n_latent)
            self.pert_embedding = None
            self.use_rdkit = False
        elif self.encoding_strategy == 'embedding':
            
            if drug_embeddings is not None:
                self.pert_embedding = drug_embeddings
                self.pert_transformation = nn.Linear(drug_embeddings.embedding_dim, self.n_latent)
                self.use_rdkit = True
            else:
                self.use_rdkit = False
                self.pert_embedding = nn.Embedding(self.n_perts, self.n_latent, padding_idx=SCAVAE_REGISTRY_KEYS.PADDING_IDX)
        elif self.encoding_strategy == 'combination_attention':
            
            self.combination_encoder = CombinationAttentionEncoder(
                n_perts=self.n_perts, n_latent=self.n_latent, num_heads=8, dropout_rate=0.1, drug_embeddings=drug_embeddings)
        else:
            raise ValueError(f"Unsupported encoding strategy: {self.encoding_strategy}")

    def _init_dose_modeling(self, n_hidden, n_layers, dropout_rate):
        
        if self.dose_aware and self.doser_type != 'none':
            if self.doser_type == 'linear':
                self.dosers = nn.Parameter(torch.ones(self.n_perts))
            elif self.doser_type == 'logsigm':
                self.dosers = GeneralizedSigmoid(self.n_perts, 'logsigm')
            elif self.doser_type == 'mlp':
                self.dosers = nn.ModuleList([
                    FCLayers(n_in=1, n_out=1, n_hidden=n_hidden or 64, n_layers=n_layers or 2,
                            use_batch_norm=False, use_layer_norm=True, dropout_rate=dropout_rate)
                    for _ in range(self.n_perts)
                ])
            else:
                
                import warnings
                warnings.warn(
                    f"Unsupported dose modeling type {self.doser_type!r}; falling back to 'linear'.",
                    RuntimeWarning,
                )
                self.doser_type = 'linear'
                self.dosers = nn.Parameter(torch.ones(self.n_perts))
        else:
            self.dosers = nn.Identity()

    def _get_base_embeddings(self, perts):
        
        bs, max_comb_len = perts.shape
        
        if self.encoding_strategy == 'onehot':
            
            pert_onehot = F.one_hot(perts, num_classes=self.n_perts).float()
            pert_onehot_sum = pert_onehot.sum(dim=1)  
            return self.pert_encoder(pert_onehot_sum)  
        
        elif self.encoding_strategy == 'combination_attention':
            
            combination_embeddings, _ = self.combination_encoder(perts)
            return combination_embeddings
        
        else:
            
            pert_embeddings = self.pert_embedding(perts)  
            
            if self.use_rdkit:
                pert_embeddings = self.pert_transformation(
                    pert_embeddings.view(bs * max_comb_len, -1)).view(bs, max_comb_len, -1)
            
            
            if self.encoding_strategy == 'learned_combination':
                weights = self.combination_weights[perts].unsqueeze(-1)
                pert_embeddings = pert_embeddings * weights
            elif self.encoding_strategy == 'attention':
                padding_mask = (perts == SCAVAE_REGISTRY_KEYS.PADDING_IDX)
                pert_embeddings, _ = self.attention(pert_embeddings, pert_embeddings, pert_embeddings, key_padding_mask=padding_mask)
            
            return pert_embeddings

    def _apply_dose_modeling(self, embeddings, perts, dosages):
        
        if not self.dose_aware or isinstance(self.dosers, nn.Identity) or dosages is None:
            return embeddings
        
        bs, max_comb_len = perts.shape
        
        if self.doser_type == 'linear':
            dose_scales = self.dosers[perts] * dosages
            return embeddings * dose_scales.unsqueeze(-1)
        
        elif self.doser_type == 'logsigm':
            dose_scales = self.dosers(dosages, perts)
            if len(embeddings.shape) == 3:  
                return torch.einsum('bm,bme->bme', [dose_scales, embeddings])
            else:  
                return embeddings * dose_scales.unsqueeze(-1)
        
        elif self.doser_type == 'mlp':
            scaled_embeddings = []
            for i in range(max_comb_len):
                pert_ids = perts[:, i]
                doses = dosages[:, i:i+1]
                batch_scaled = []
                for pert_id, dose in zip(pert_ids, doses):
                    if pert_id != SCAVAE_REGISTRY_KEYS.PADDING_IDX:
                        scale = self.dosers[pert_id](dose.unsqueeze(0)).squeeze()
                        batch_scaled.append(scale)
                    else:
                        batch_scaled.append(torch.tensor(0.0, device=doses.device))
                dose_scales_i = torch.stack(batch_scaled)
                scaled_embeddings.append(embeddings[:, i] * dose_scales_i.unsqueeze(-1))
            return torch.stack(scaled_embeddings, dim=1)
        
        return embeddings

    def _combine_perturbations(self, embeddings, perts):
        
        
        if self.encoding_strategy == 'onehot':
            return embeddings
        
        
        padding_mask = (perts != SCAVAE_REGISTRY_KEYS.PADDING_IDX).float().unsqueeze(-1)
        masked_embeddings = embeddings * padding_mask
        return masked_embeddings.sum(dim=1)

    def forward(self, perts, dosages=None):
        
        perts = perts.long()
        
        
        embeddings = self._get_base_embeddings(perts)
        
        
        embeddings = self._apply_dose_modeling(embeddings, perts, dosages)
        
        
        final_representation = self._combine_perturbations(embeddings, perts)
        
        return final_representation

class Decoder(nn.Module):
    
    
    def __init__(
        self, 
        latent_dim, 
        n_genes, 
        n_layers=3, 
        hidden_dim=256, 
        dropout=0.1,
        activation='telu',
        use_batch_norm=True,
        use_layer_norm=False,
        output_activation= 'linear'
    ):
        super().__init__()
        
        
        self.latent_dim = latent_dim
        self.n_genes = n_genes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        
        if activation == 'relu':
            self.activation_fn = nn.ReLU
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU
        elif activation == 'gelu':
            self.activation_fn = nn.GELU
        elif activation == 'swish':
            self.activation_fn = nn.SiLU
        elif activation == 'telu':
            self.activation_fn = TeLU
        else:
            self.activation_fn = nn.ReLU

        self.net = FCLayers(
            n_in=latent_dim,
            n_out=hidden_dim,
            n_layers=n_layers,
            n_hidden=hidden_dim,
            dropout_rate=dropout,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=self.activation_fn,
        )
        self.final_projection = nn.Linear(hidden_dim, n_genes)
         
        self.output_activation = output_activation
    
    def forward(self, x):
        
        output = self.net(x)
        
        output = self.final_projection(output)
        
        if self.output_activation == 'softplus':
            
            output = F.softplus(output, beta=1, threshold=10)
        elif self.output_activation == 'relu':
            output = F.relu(output)
        elif self.output_activation == 'linear':
        
            pass   
        return output

class FocalLoss(nn.Module):
    

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ):
        
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if len(y_true) == 0:
            return torch.tensor(0.)

        
        
        log_p = F.log_softmax(y_pred, dim=-1)
        ce = self.nll_loss(log_p, y_true)

        
        all_rows = torch.arange(len(y_pred))
        log_pt = log_p[all_rows, y_true]

        
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
