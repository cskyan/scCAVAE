import math

from collections import defaultdict
from typing import Union

import torch
from scvi.module import Classifier
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from scvi.train import TrainingPlan

import numpy as np
from torchmetrics.functional import accuracy

from ._module import scCAVAEModule
from ._utils import SCAVAE_REGISTRY_KEYS, FocalLoss
from typing import Optional

class scCAVAETrainingPlan(TrainingPlan):
    def __init__(
            self,
            module: scCAVAEModule,
            covars_to_ncovars: dict,
            n_adv_perts: int,
            lr=5e-4,
            wd=1e-6,
            n_steps_pretrain_ae: Optional[int] = None,
            n_epochs_pretrain_ae: Optional[int] = None,
            n_steps_kl_warmup: Optional[int] = None,
            n_epochs_kl_warmup: Optional[int] = None,
            n_steps_adv_warmup: Optional[int] = None,
            n_epochs_adv_warmup: Optional[int] = None,
            n_epochs_mixup_warmup: Optional[int] = None,
            n_epochs_verbose: Optional[int] = 10,
            mixup_alpha: float = 0.0,
            adv_steps: int = 3,
            reg_adv: float = 1.,
            pen_adv: float = 1.,
            n_hidden_adv: int = 64,
            n_layers_adv: int = 3,
            use_batch_norm_adv: bool = True,
            use_layer_norm_adv: bool = False,
            dropout_rate_adv: float = 0.1,
            adv_lr=3e-4,
            adv_wd=4e-7,
            doser_lr=3e-4,
            doser_wd=4e-7,
            step_size_lr: Optional[int] = 45,
            do_clip_grad: Optional[bool] = False,
            gradient_clip_value: Optional[float] = 3.0,
            drug_weights: Optional[list] = None,
            adv_loss: Optional[str] = 'cce',
            use_adversarial_training: bool = True,
            contrastive_loss_type: str = 'none',
            reg_contrastive: float = 0.1,  
            n_hard_negatives: int = 5,
            n_hard_positives: int = 3,
            use_cell_type_contrast: bool = True,
            use_global_hierarchical: bool = False,  
            supcon_temperature: float = 0.1,  
            infonce_temperature: float = 0.15,
            multilevel_contrastive: bool = True,
            pert_contrastive_weight: float = 1.0,
            full_contrastive_weight: float = 0.0,
            false_negative_threshold: float = 0.85,  
            n_epochs_contrastive_warmup: Optional[int] = None,
            n_steps_contrastive_warmup: Optional[int] = None,
    ):
        
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=wd,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=False,
            lr_factor=None,
            lr_patience=None,
            lr_threshold=None,
            lr_scheduler_metric=None,
            lr_min=None,
        )

        self.automatic_optimization = False

        self.wd = wd

        self.covars_encoder = covars_to_ncovars

        self.mixup_alpha = mixup_alpha
        self.n_epochs_mixup_warmup = n_epochs_mixup_warmup

        self.n_steps_pretrain_ae = n_steps_pretrain_ae
        self.n_epochs_pretrain_ae = n_epochs_pretrain_ae

        self.n_steps_adv_warmup = n_steps_adv_warmup
        self.n_epochs_adv_warmup = n_epochs_adv_warmup
        
        self.n_epochs_contrastive_warmup = n_epochs_contrastive_warmup

        self.n_epochs_verbose = n_epochs_verbose

        self.adv_steps = adv_steps

        self.reg_adv = reg_adv
        self.pen_adv = pen_adv

        self.adv_lr = adv_lr
        self.adv_wd = adv_wd

        self.doser_lr = doser_lr
        self.doser_wd = doser_wd

        self.step_size_lr = step_size_lr

        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value

        self.metrics = ['train_loss', 'recon_loss', 'KL',
                        'disnt_basal', 'disnt_after',
                        'r2_mean', 'r2_var', 'scavae_metric',
                        'adv_loss', 'penalty_adv', 'adv_perts', 'acc_perts', 'penalty_perts',
                        'contrastive_loss',
                        ]

        self.epoch_history = defaultdict(list)
        self.n_adv_perts = n_adv_perts
        
        self.use_adversarial_training = use_adversarial_training
        
        
        if self.use_adversarial_training:
            self.perturbation_classifier = Classifier(
                n_input=self.module.n_latent,
                n_labels=n_adv_perts,
                n_hidden=n_hidden_adv,
                n_layers=n_layers_adv,
                use_batch_norm=use_batch_norm_adv,
                use_layer_norm=use_layer_norm_adv,
                dropout_rate=dropout_rate_adv,
                activation_fn=nn.ReLU,
                logits=True,
            )

            self.covars_classifiers = nn.ModuleDict(
                {
                    key: Classifier(n_input=self.module.n_latent,
                                    n_labels=len(unique_covars),
                                    n_hidden=n_hidden_adv,
                                    n_layers=n_layers_adv,
                                    use_batch_norm=use_batch_norm_adv,
                                    use_layer_norm=use_layer_norm_adv,
                                    dropout_rate=dropout_rate_adv,
                                    logits=True)
                    if len(unique_covars) > 1 else None

                    for key, unique_covars in self.covars_encoder.items()
                }
            )
            
            self.drug_weights = torch.tensor(drug_weights).to(self.device) if drug_weights else torch.ones(
                self.n_adv_perts).to(self.device)

            self.adv_loss = adv_loss.lower()
            self.gamma = 2.0
            if self.adv_loss == 'cce':
                self.adv_loss_drugs = nn.CrossEntropyLoss(weight=self.drug_weights)
                self.adv_loss_fn = nn.CrossEntropyLoss()
            elif self.adv_loss == 'focal':
                self.adv_loss_drugs = FocalLoss(alpha=self.drug_weights, gamma=self.gamma, reduction='mean')
                self.adv_loss_fn = FocalLoss(gamma=self.gamma, reduction='mean')
        else:
            
            self.perturbation_classifier = None
            self.covars_classifiers = None
            self.drug_weights = None
            self.adv_loss_drugs = None
            self.adv_loss_fn = None

        self.n_steps_contrastive_warmup = n_steps_contrastive_warmup
        self.n_epochs_contrastive_warmup = n_epochs_contrastive_warmup        

        self.contrastive_loss_type = contrastive_loss_type.lower()
        self.n_hard_negatives = n_hard_negatives
        self.reg_contrastive = reg_contrastive
        self.supcon_temperature = supcon_temperature
        self.infonce_temperature = infonce_temperature
        self.multilevel_contrastive = multilevel_contrastive
        self.pert_contrastive_weight = pert_contrastive_weight
        self.full_contrastive_weight = full_contrastive_weight
        self.n_hard_positives = n_hard_positives
        self.use_cell_type_contrast = use_cell_type_contrast
        self.use_global_hierarchical = use_global_hierarchical
        self.false_negative_threshold = false_negative_threshold
    @property
    def adv_lambda(self):
        slope = self.reg_adv
        if self.n_steps_adv_warmup:
            global_step = self.global_step

            if self.n_steps_pretrain_ae:
                 global_step -= self.n_steps_pretrain_ae

            if global_step <= self.n_steps_adv_warmup:
                proportion = global_step / self.n_steps_adv_warmup
                return slope * proportion
            else:
                return slope
        elif self.n_epochs_adv_warmup is not None:
            current_epoch = self.current_epoch

            if self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            if current_epoch <= self.n_epochs_adv_warmup:
                proportion = current_epoch / self.n_epochs_adv_warmup
                return slope * proportion
            else:
                return slope
        else:
            return slope

    @property
    def alpha_mixup(self):
        if self.n_epochs_mixup_warmup:
            current_epoch = self.current_epoch

            if self.n_epochs_pretrain_ae:
                current_epoch -= self.current_epoch

            if current_epoch <= self.n_epochs_mixup_warmup:
                proportion = current_epoch / self.n_epochs_mixup_warmup

                return self.mixup_alpha * proportion
            else:
                return self.mixup_alpha
        else:
            return self.mixup_alpha

    @property
    def do_start_adv_training(self):
        
        if not self.use_adversarial_training:
            return False
            
        if self.n_steps_pretrain_ae:
            return self.global_step > self.n_steps_pretrain_ae
        elif self.n_epochs_pretrain_ae:
            return self.current_epoch > self.n_epochs_pretrain_ae
        else:
            return True

    def adversarial_loss(self, tensors, z_basal, mixup_lambda: float = 1.0, compute_penalty=True):
        
        
        if not self.use_adversarial_training:
            dummy_tensor = torch.tensor(0.0).to(self.device)
            adv_results = {
                'adv_loss': dummy_tensor,
                'penalty_adv': dummy_tensor,
                'adv_perts': dummy_tensor,
                'acc_perts': dummy_tensor,
                'penalty_perts': dummy_tensor
            }
            
            for covar in self.covars_encoder.keys():
                adv_results[f'adv_{covar}'] = dummy_tensor
                adv_results[f'acc_{covar}'] = dummy_tensor
                adv_results[f'penalty_{covar}'] = dummy_tensor
            
            return adv_results

        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        covars_dict = dict()
        for covar, unique_covars in self.covars_encoder.items():
            encoded_covars = tensors[covar].view(-1, )  
            covars_dict[covar] = encoded_covars

        covars_pred = {}
        for covar in self.covars_encoder.keys():
            if self.covars_classifiers[covar] is not None:
                covar_pred = self.covars_classifiers[covar](z_basal)
                covars_pred[covar] = covar_pred
            else:
                covars_pred[covar] = None

        adv_results = {}

        
        for covar, covars in self.covars_encoder.items():
            adv_results[f'adv_{covar}'] = mixup_lambda * self.adv_loss_fn(
                covars_pred[covar],
                covars_dict[covar].long(),
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device) + (
                    1. - mixup_lambda) * self.adv_loss_fn(
                covars_pred[covar],
                covars_dict[covar + '_mixup'].long(),
            ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)
            adv_results[f'acc_{covar}'] = accuracy(
                covars_pred[covar].argmax(1), covars_dict[covar].long(), task='multiclass',
                num_classes=len(covars))                if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

        adv_results['adv_loss'] = sum([adv_results[f'adv_{key}'] for key in self.covars_encoder.keys()])

        perturbations = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY].view(-1, )
        perturbations_mixup = tensors[SCAVAE_REGISTRY_KEYS.PERTURBATION_KEY + '_mixup'].view(-1, )

        perturbations_pred = self.perturbation_classifier(z_basal)

        adv_results['adv_perts'] = mixup_lambda * self.adv_loss_drugs(perturbations_pred,
                                                                      perturbations.long()) + (
                                           1. - mixup_lambda) * self.adv_loss_drugs(perturbations_pred,
                                                                                    perturbations_mixup.long())

        adv_results['acc_perts'] = mixup_lambda * accuracy(
            perturbations_pred.argmax(1), perturbations.long().view(-1, ), average='macro',
            num_classes=self.n_adv_perts, task='multiclass',
        ) + (1. - mixup_lambda) * accuracy(
            perturbations_pred.argmax(1), perturbations_mixup.long().view(-1, ), average='macro',
            num_classes=self.n_adv_perts, task='multiclass',
        )

        adv_results['adv_loss'] += adv_results['adv_perts']

        if compute_penalty:
            
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = (
                    torch.autograd.grad(
                        covars_pred[covar].sum(),
                        z_basal,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0].pow(2).mean()
                ) if covars_pred[covar] is not None else torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_adv'] = sum([adv_results[f'penalty_{covar}'] for covar in self.covars_encoder.keys()])

            adv_results['penalty_perts'] = (
                torch.autograd.grad(
                    perturbations_pred.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0].pow(2).mean()
            )

            adv_results['penalty_adv'] += adv_results['penalty_perts']
        else:
            for covar in self.covars_encoder.keys():
                adv_results[f'penalty_{covar}'] = torch.as_tensor(0.0).to(self.device)

            adv_results['penalty_perts'] = torch.as_tensor(0.0).to(self.device)
            adv_results['penalty_adv'] = torch.as_tensor(0.0).to(self.device)

        return adv_results
    
    def configure_optimizers(self):
        
        pert_params = []

        
        if hasattr(self.module.pert_network, 'pert_embedding') and self.module.pert_network.pert_embedding is not None:
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.pert_embedding.parameters()))
        elif hasattr(self.module.pert_network, 'combination_encoder'):
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.combination_encoder.parameters()))
        elif hasattr(self.module.pert_network, 'combination_token_encoder'):
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.combination_token_encoder.parameters()))
        elif hasattr(self.module.pert_network, 'combination_token_self_attn_encoder'):
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.combination_token_self_attn_encoder.parameters()))
        elif hasattr(self.module.pert_network, 'pert_encoder'):
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.pert_encoder.parameters()))
        else:
            
            pert_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.parameters()))

        ae_params = list(filter(lambda p: p.requires_grad, self.module.encoder.parameters())) +                    list(filter(lambda p: p.requires_grad, self.module.decoder.parameters())) +                    pert_params +                    list(filter(lambda p: p.requires_grad, self.module.covars_embeddings.parameters()))

        if self.module.recon_loss in ['zinb', 'nb']:
            ae_params += [self.module.px_r]

        optimizer_autoencoder = torch.optim.AdamW(
            ae_params,
            lr=self.lr,
            weight_decay=self.wd)

        
        scheduler_autoencoder = {
            'scheduler': ReduceLROnPlateau(optimizer_autoencoder, mode='max', factor=0.5, patience=10),
            'monitor': 'val_r2_mean', 
            'interval': 'epoch',
            'frequency': 1
        }
        
        doser_params = []
        if hasattr(self.module.pert_network, 'dosers') and self.module.pert_network.dosers is not None:
            
            if not isinstance(self.module.pert_network.dosers, torch.nn.Identity):
                
                if isinstance(self.module.pert_network.dosers, torch.nn.Parameter):
                    
                    if self.module.pert_network.dosers.requires_grad:
                        doser_params = [self.module.pert_network.dosers]
                elif hasattr(self.module.pert_network.dosers, 'parameters'):
                    
                    doser_params = list(filter(lambda p: p.requires_grad, self.module.pert_network.dosers.parameters()))
                else:
                    pass
            else:
                pass

        
        if not doser_params:
            doser_params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

        optimizer_doser = torch.optim.AdamW(
            doser_params, lr=self.doser_lr, weight_decay=self.doser_wd,
        )
        
        scheduler_doser = {
            'scheduler': ReduceLROnPlateau(optimizer_doser, mode='max', factor=0.5, patience=10),
            'monitor': 'val_r2_mean', 
            'interval': 'epoch',
            'frequency': 1
        }
        
        optimizers = [optimizer_autoencoder, optimizer_doser]
        schedulers = [scheduler_autoencoder, scheduler_doser]
        
        
        if self.use_adversarial_training:
            adv_params = list(filter(lambda p: p.requires_grad, self.perturbation_classifier.parameters())) +                        list(filter(lambda p: p.requires_grad, self.covars_classifiers.parameters()))

            optimizer_adversaries = torch.optim.AdamW(
                adv_params,
                lr=self.adv_lr,
                weight_decay=self.adv_wd)
            scheduler_adversaries = StepLR(optimizer_adversaries, step_size=self.step_size_lr, gamma=0.9)
            
            optimizers.append(optimizer_adversaries)
            schedulers.append(scheduler_adversaries)
        else:
            
            
            optimizer_adversaries = torch.optim.AdamW(
                [nn.Parameter(torch.zeros(1))],  
                lr=self.adv_lr)
            scheduler_adversaries = StepLR(optimizer_adversaries, step_size=self.step_size_lr, gamma=0.9)
            
            optimizers.append(optimizer_adversaries)
            schedulers.append(scheduler_adversaries)

        if self.step_size_lr is not None:
            return optimizers, schedulers
        else:
            return optimizers

    @property
    def contrastive_lambda(self):
        
        slope = self.reg_contrastive

        
        if self.contrastive_loss_type == 'none':
            return 0.0

        
        if self.n_steps_contrastive_warmup is not None:
            global_step = self.global_step

            
            if self.n_steps_pretrain_ae:
                global_step -= self.n_steps_pretrain_ae

            
            if global_step <= 0:
                return 0.0
            elif global_step <= self.n_steps_contrastive_warmup:
                proportion = global_step / self.n_steps_contrastive_warmup
                return slope * proportion
            else:
                return slope

        
        elif self.n_epochs_contrastive_warmup is not None:
            current_epoch = self.current_epoch

            
            if self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            
            if current_epoch <= 0:
                return 0.0
            elif current_epoch <= self.n_epochs_contrastive_warmup:
                proportion = current_epoch / self.n_epochs_contrastive_warmup
                return slope * proportion
            else:
                return slope
        else:
            
            if self.n_steps_pretrain_ae and self.global_step <= self.n_steps_pretrain_ae:
                return 0.0
            elif self.n_epochs_pretrain_ae and self.current_epoch <= self.n_epochs_pretrain_ae:
                return 0.0
            else:
                return slope

    @property
    def do_start_contrastive_training(self):
        
        
        if self.contrastive_loss_type == 'none':
            return False
            
        
        if self.n_steps_pretrain_ae:
            return self.global_step > self.n_steps_pretrain_ae
        elif self.n_epochs_pretrain_ae:
            return self.current_epoch > self.n_epochs_pretrain_ae
        else:
            
            return True

    def training_step(self, batch, batch_idx):
        opt, opt_doser, opt_adv = self.optimizers()
        mixup_alpha = self.alpha_mixup
        batch, mixup_lambda = self.module.mixup_data(batch, alpha=mixup_alpha)
        
        
        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False,
                                                      inference_kwargs={'mixup_lambda': mixup_lambda, 'compute_all_latents': False})
        
        
        
        recon_loss, kl_loss, contrastive_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
            contrastive_loss_type=self.contrastive_loss_type,  
            supcon_temperature=self.supcon_temperature,
            n_hard_negatives=self.n_hard_negatives,
            n_hard_positives=self.n_hard_positives,
            use_cell_type_contrast=self.use_cell_type_contrast,
            use_global_hierarchical=self.use_global_hierarchical,
            infonce_temperature=self.infonce_temperature,
            multilevel_contrastive=self.multilevel_contrastive,
            pert_contrastive_weight=self.pert_contrastive_weight,
            full_contrastive_weight=self.full_contrastive_weight,
            false_negative_threshold=self.false_negative_threshold,
        )
        
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        
        if self.contrastive_loss_type != 'none':
            total_loss += self.contrastive_lambda * contrastive_loss
        
        z_basal = inf_outputs['z_basal']
        
        if self.use_adversarial_training and self.do_start_adv_training:
            
            if self.adv_steps is None:
                
                opt.zero_grad()
                opt_doser.zero_grad()
                
                adv_results = self.adversarial_loss(tensors=batch,
                                                z_basal=z_basal,
                                                mixup_lambda=mixup_lambda,
                                                compute_penalty=False)
                
                
                total_loss = recon_loss + self.kl_weight * kl_loss
                if self.contrastive_loss_type != 'none':
                    total_loss += self.contrastive_lambda * contrastive_loss
                total_loss -= self.adv_lambda * adv_results['adv_loss']  
                
                self.manual_backward(total_loss)
                
                if self.do_clip_grad:
                    self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                    self.clip_gradients(opt_doser, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                
                opt.step()
                opt_doser.step()
                
                
                opt_adv.zero_grad()
                adv_results = self.adversarial_loss(tensors=batch,
                                                z_basal=z_basal.detach(),
                                                mixup_lambda=mixup_lambda,
                                                compute_penalty=True)
                adv_loss = adv_results['adv_loss'] + self.pen_adv * adv_results['penalty_adv']
                self.manual_backward(adv_loss)
                
                if self.do_clip_grad:
                    self.clip_gradients(opt_adv, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                opt_adv.step()
                
            elif batch_idx % self.adv_steps != 0:
                
                opt_adv.zero_grad()
                adv_results = self.adversarial_loss(tensors=batch,
                                                z_basal=z_basal.detach(),
                                                mixup_lambda=mixup_lambda,
                                                compute_penalty=True)
                adv_loss = adv_results['adv_loss'] + self.pen_adv * adv_results['penalty_adv']
                self.manual_backward(adv_loss)
                
                if self.do_clip_grad:
                    self.clip_gradients(opt_adv, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                opt_adv.step()
                
            else:
                
                opt.zero_grad()
                opt_doser.zero_grad()
                
                adv_results = self.adversarial_loss(tensors=batch,
                                                z_basal=z_basal,
                                                mixup_lambda=mixup_lambda,
                                                compute_penalty=False)
                
                total_loss = recon_loss + self.kl_weight * kl_loss
                if self.contrastive_loss_type != 'none':
                    total_loss += self.contrastive_lambda * contrastive_loss
                total_loss -= self.adv_lambda * adv_results['adv_loss']  
                
                self.manual_backward(total_loss)
                
                if self.do_clip_grad:
                    self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                    self.clip_gradients(opt_doser, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                opt.step()
                opt_doser.step()
                
        else:
            
            opt.zero_grad()
            opt_doser.zero_grad()
            
            total_loss = recon_loss + self.kl_weight * kl_loss
            if self.contrastive_loss_type != 'none':
                total_loss += self.contrastive_lambda * contrastive_loss
            
            self.manual_backward(total_loss)
            
            if self.do_clip_grad:
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
                self.clip_gradients(opt_doser, gradient_clip_val=self.gradient_clip_value, gradient_clip_algorithm="norm")
            
            opt.step()
            opt_doser.step()
            
            
            adv_results = self.adversarial_loss(tensors=batch, z_basal=z_basal, mixup_lambda=mixup_lambda)
        
        
        
        
        
        
        for key, val in adv_results.items():
            adv_results[key] = val.detach() if hasattr(val, 'detach') else val

        results = adv_results.copy()
        results.update({'recon_loss': recon_loss.detach()})
        results.update({'KL': kl_loss.detach()})

        
        if self.contrastive_loss_type != "none":
            results.update({'contrastive_loss': contrastive_loss.detach()})
        else:
            results.update({'contrastive_loss': 0.0})

        return results

    def training_epoch_end(self, outputs):
        for key in self.metrics:
            if key in ['disnt_basal', 'disnt_after']:
                self.epoch_history[key].append(0.0)
            else:
                self.epoch_history[key].append(np.mean([output[key] for output in outputs if key in output and output[key] != 0.0]))

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                key1, key2, key3 = f'adv_{covar}', f'penalty_{covar}', f'acc_{covar}'
                self.epoch_history[key1].append(np.mean([output[key1] for output in outputs if key1 in output and output[key1] != 0.0]))
                self.epoch_history[key2].append(np.mean([output[key2] for output in outputs if key2 in output and output[key2] != 0.0]))
                self.epoch_history[key3].append(np.mean([output[key3] for output in outputs if key3 in output and output[key3] != 0.0]))

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('train')

        self.log("recon", self.epoch_history['recon_loss'][-1], prog_bar=True)
        

        if self.use_adversarial_training:
            self.log("adv_loss", self.epoch_history['adv_loss'][-1], prog_bar=True)
            self.log("acc_pert", self.epoch_history['acc_perts'][-1], prog_bar=True)
            for covar, nc in self.covars_encoder.items():
                if len(nc) > 1:
                    self.log(f'acc_{covar}', self.epoch_history[f'acc_{covar}'][-1], prog_bar=True)

        if self.contrastive_loss_type != 'none' and 'contrastive_loss' in self.epoch_history:
            
            self.log("contrastive", self.epoch_history['contrastive_loss'][-1], prog_bar=True)

        
        

        
        

    def validation_step(self, batch, batch_idx):
        
        batch, mixup_lambda = self.module.mixup_data(batch, alpha=0.0)  

        inf_outputs, gen_outputs = self.module.forward(batch, compute_loss=False,
                                                       inference_kwargs={
                                                           'mixup_lambda': 1.0,
                                                           'compute_all_latents': False,
                                                       })

        
        recon_loss, kl_loss, contrastive_loss = self.module.loss(
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
            contrastive_loss_type=self.contrastive_loss_type,  
            supcon_temperature=self.supcon_temperature,
            n_hard_negatives=self.n_hard_negatives,
            n_hard_positives=self.n_hard_positives,
            use_cell_type_contrast=self.use_cell_type_contrast,
            use_global_hierarchical=self.use_global_hierarchical,
            infonce_temperature=self.infonce_temperature,
            multilevel_contrastive=self.multilevel_contrastive,
            pert_contrastive_weight=self.pert_contrastive_weight,
            full_contrastive_weight=self.full_contrastive_weight,
            false_negative_threshold=self.false_negative_threshold,
        )

        total_loss = recon_loss + self.kl_weight * kl_loss

        
        if self.contrastive_loss_type != 'none':
            total_loss += self.contrastive_lambda * contrastive_loss
        
        self.log("val_loss", total_loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        self.log("val_contrastive", contrastive_loss)

        
        z_basal = inf_outputs['z_basal']
        adv_results = self.adversarial_loss(
            tensors=batch,
            z_basal=z_basal,
            mixup_lambda=1.0, 
            compute_penalty=False 
        )

        for key, val in adv_results.items():
            adv_results[key] = val.detach() if hasattr(val, 'detach') else val

        results = adv_results.copy()

        
        r2_mean, r2_var = self.module.r2_metric(batch, inf_outputs, gen_outputs, mode='direct')
        disnt_basal, disnt_after = self.module.disentanglement(batch, inf_outputs, gen_outputs)

        results.update({'r2_mean': r2_mean, 'r2_var': r2_var})
        results.update({'disnt_basal': disnt_basal})
        results.update({'disnt_after': disnt_after})
        results.update({'KL': kl_loss.detach()})
        results.update({'recon_loss': recon_loss.detach()})
        if self.contrastive_loss_type != "none":
            results.update({'contrastive_loss': contrastive_loss.detach()})
        else:
            results.update({'contrastive_loss': 0.0})
        
        
        disnt_alpha = 0.1
        disnt_diff = disnt_after - disnt_basal
        if self.module.recon_loss in ['mse_sum', 'mse', 'huber']:
            results.update({'scavae_metric': r2_mean + disnt_alpha * disnt_diff})
        else:
            results.update({'scavae_metric': r2_mean + 0.5 * r2_var + disnt_alpha * disnt_diff})

        return results

    
    def validation_epoch_end(self, outputs):
        for key in self.metrics:
            
            values = [output[key] for output in outputs if key in output and output[key] != 0.0]
            if values:
                self.epoch_history[key].append(np.mean(values))
            else:
                
                self.epoch_history[key].append(0.0)

        for covar, unique_covars in self.covars_encoder.items():
            if len(unique_covars) > 1:
                key1, key2, key3 = f'adv_{covar}', f'penalty_{covar}', f'acc_{covar}'
                
                values1 = [output[key1] for output in outputs if key1 in output and output[key1] != 0.0]
                values2 = [output[key2] for output in outputs if key2 in output and output[key2] != 0.0]  
                values3 = [output[key3] for output in outputs if key3 in output and output[key3] != 0.0]
                
                self.epoch_history[key1].append(np.mean(values1) if values1 else 0.0)
                self.epoch_history[key2].append(np.mean(values2) if values2 else 0.0)
                self.epoch_history[key3].append(np.mean(values3) if values3 else 0.0)

        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('valid')

        self.log('val_recon', self.epoch_history['recon_loss'][-1], prog_bar=True)
        self.log('disnt_basal', self.epoch_history['disnt_basal'][-1], prog_bar=True)
        self.log('disnt_after', self.epoch_history['disnt_after'][-1], prog_bar=True)
        self.log('val_r2_mean', self.epoch_history['r2_mean'][-1], prog_bar=True)
        self.log('val_r2_var', self.epoch_history['r2_var'][-1], prog_bar=False)
        self.log('val_KL', self.epoch_history['KL'][-1], prog_bar=True)
        
        
        
        if self.module.recon_loss in ['mse', 'mse_sum', 'huber']:
            scavae_metric = np.mean([output['r2_mean'] for output in outputs if 'r2_mean' in output])
            self.log('scavae_metric', scavae_metric, prog_bar=True)
        else:
            
            self.log('scavae_metric', np.mean([output["scavae_metric"] for output in outputs if 'scavae_metric' in output]), prog_bar=True)
        
        if self.current_epoch % self.n_epochs_verbose == self.n_epochs_verbose - 1:
            print(f'\ndisnt_basal = {self.epoch_history["disnt_basal"][-1]}')
            print(f'disnt_after = {self.epoch_history["disnt_after"][-1]}')
            print(f'val_r2_mean = {self.epoch_history["r2_mean"][-1]}')
            print(f'val_r2_var = {self.epoch_history["r2_var"][-1]}')

        
        if self.contrastive_loss_type != "none":
            contrastive_values = [output['contrastive_loss'] for output in outputs if 'contrastive_loss' in output]
            if contrastive_values:
                val_contrastive = np.mean(contrastive_values)
                
                self.log('val_contrastive', val_contrastive, prog_bar=True)
                self.epoch_history['val_contrastive'] = val_contrastive
                
                print(f'val_contrastive = {val_contrastive}')
        
        
        scavae_values = [output["scavae_metric"] for output in outputs if 'scavae_metric' in output]
        if scavae_values:
            print(f'scavae_metric = {np.mean(scavae_values)}')
