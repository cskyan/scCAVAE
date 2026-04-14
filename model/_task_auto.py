import torch
from torch.optim.lr_scheduler import StepLR
from scvi.train import TrainingPlan

class scCAVAEAutoTrainingPlan(TrainingPlan):
    

    def __init__(
        self,
        module,
        covars_to_ncovars=None,  
        lr=5e-4,
        wd=1e-6,
        step_size_lr=45,
        lr_gamma=0.9,
        optimizer_type="adamw",
        **kwargs  
    ):
        
        
        super().__init__(module=module)

        
        self.automatic_optimization = True

        
        self.lr = lr
        self.wd = wd
        self.step_size_lr = step_size_lr
        self.lr_gamma = lr_gamma
        self.optimizer_type = optimizer_type

        
        for key, value in kwargs.items():
            setattr(self, key, value)

        
        from collections import defaultdict
        self.epoch_history = defaultdict(list)

    @property
    def alpha_mixup(self):
        
        if hasattr(self, 'n_epochs_mixup_warmup') and self.n_epochs_mixup_warmup:
            current_epoch = self.current_epoch

            if hasattr(self, 'n_epochs_pretrain_ae') and self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            if current_epoch <= self.n_epochs_mixup_warmup:
                proportion = current_epoch / self.n_epochs_mixup_warmup
                return getattr(self, 'mixup_alpha', 0.0) * proportion
            else:
                return getattr(self, 'mixup_alpha', 0.0)
        else:
            return getattr(self, 'mixup_alpha', 0.0)

    @property
    def contrastive_lambda(self):
        
        slope = getattr(self, 'reg_contrastive', 0.1)

        
        if getattr(self, 'contrastive_loss_type', 'none') == 'none':
            return 0.0

        
        if hasattr(self, 'n_steps_contrastive_warmup') and self.n_steps_contrastive_warmup is not None:
            global_step = self.global_step

            
            if hasattr(self, 'n_steps_pretrain_ae') and self.n_steps_pretrain_ae:
                global_step -= self.n_steps_pretrain_ae

            
            if global_step <= 0:
                return 0.0
            elif global_step <= self.n_steps_contrastive_warmup:
                proportion = global_step / self.n_steps_contrastive_warmup
                return slope * proportion
            else:
                return slope

        
        elif hasattr(self, 'n_epochs_contrastive_warmup') and self.n_epochs_contrastive_warmup is not None:
            current_epoch = self.current_epoch

            
            if hasattr(self, 'n_epochs_pretrain_ae') and self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            
            if current_epoch <= 0:
                return 0.0
            elif current_epoch <= self.n_epochs_contrastive_warmup:
                proportion = current_epoch / self.n_epochs_contrastive_warmup
                return slope * proportion
            else:
                return slope
        else:
            
            if (hasattr(self, 'n_steps_pretrain_ae') and self.n_steps_pretrain_ae and
                self.global_step <= self.n_steps_pretrain_ae):
                return 0.0
            elif (hasattr(self, 'n_epochs_pretrain_ae') and self.n_epochs_pretrain_ae and
                  self.current_epoch <= self.n_epochs_pretrain_ae):
                return 0.0
            else:
                return slope

    def configure_optimizers(self):
        
        
        
        if getattr(self, 'optimizer_type', 'adamw').lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.module.parameters(),
                lr=self.lr,          
                weight_decay=self.wd 
            )
        else:  
            optimizer = torch.optim.Adam(
                self.module.parameters(),
                lr=self.lr,          
                weight_decay=self.wd 
            )

        
        scheduler_type = getattr(self, 'scheduler_type', 'none').lower()

        if scheduler_type == 'plateau':
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  
                factor=0.5,  
                patience=10,  
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_r2_mean',  
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        elif scheduler_type == 'step':
            
            step_size = getattr(self, 'step_size_lr', 30)
            gamma = getattr(self, 'lr_gamma', 0.9)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return [optimizer], [scheduler]

        else:
            
            return optimizer

    def training_step(self, batch, batch_idx):
        

        
        mixup_alpha = self.alpha_mixup
        batch, mixup_lambda = self.module.mixup_data(batch, alpha=mixup_alpha)

        
        inf_outputs, gen_outputs = self.module.forward(
            batch,
            compute_loss=False,
            inference_kwargs={'mixup_lambda': mixup_lambda, 'compute_all_latents': False}
        )

        
        recon_loss, kl_loss, contrastive_loss = self.module.loss(  
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
            contrastive_loss_type=getattr(self, 'contrastive_loss_type', 'none'),
            supcon_temperature=getattr(self, 'supcon_temperature', 0.1),
            n_hard_negatives=getattr(self, 'n_hard_negatives', 5),
            n_hard_positives=getattr(self, 'n_hard_positives', 3),
            use_cell_type_contrast=getattr(self, 'use_cell_type_contrast', True),
            infonce_temperature=getattr(self, 'infonce_temperature', 0.15),
            multilevel_contrastive=getattr(self, 'multilevel_contrastive', True),
            pert_contrastive_weight=getattr(self, 'pert_contrastive_weight', 1.0),
            full_contrastive_weight=getattr(self, 'full_contrastive_weight', 0.0),
            false_negative_threshold=getattr(self, 'false_negative_threshold', 0.85),
        )

        
        if contrastive_loss is None:
            contrastive_loss = recon_loss.new_tensor(0.0)

        
        total_loss = recon_loss + self.kl_weight * kl_loss

        if getattr(self, 'contrastive_loss_type', 'none') != 'none':
            total_loss += self.contrastive_lambda * contrastive_loss

        
        
        self.log("total_loss", total_loss, prog_bar=True, sync_dist=False)
        self.log("recon", recon_loss, prog_bar=True, sync_dist=False)
        self.log("kl", kl_loss, prog_bar=True, sync_dist=False)
        if getattr(self, 'contrastive_loss_type', 'none') != 'none':
            self.log("contrastive", contrastive_loss, prog_bar=True, sync_dist=False)

        
        self._last_train_loss = total_loss.detach()
        self._last_recon_loss = recon_loss.detach()
        self._last_kl_loss = kl_loss.detach()
        self._last_contrastive_loss = contrastive_loss.detach() if hasattr(contrastive_loss, "detach") else contrastive_loss

        
        return total_loss

    def training_epoch_end(self, outputs):
        
        
        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('train')

        
        self.epoch_history['train_loss'].append(float(getattr(self, '_last_train_loss', 0.0)))
        self.epoch_history['recon_loss'].append(float(getattr(self, '_last_recon_loss', 0.0)))
        self.epoch_history['KL'].append(float(getattr(self, '_last_kl_loss', 0.0)))
        self.epoch_history['contrastive_loss'].append(float(getattr(self, '_last_contrastive_loss', 0.0)))

        
        self.epoch_history['r2_mean'].append(0.0)
        self.epoch_history['r2_var'].append(0.0)
        self.epoch_history['disnt_basal'].append(0.0)
        self.epoch_history['disnt_after'].append(0.0)
        self.epoch_history['scavae_metric'].append(0.0)

    def validation_step(self, batch, batch_idx):  
        

        
        batch, mixup_lambda = self.module.mixup_data(batch, alpha=0.0)

        
        inf_outputs, gen_outputs = self.module.forward(
            batch,
            compute_loss=False,
            inference_kwargs={'mixup_lambda': 1.0, 'compute_all_latents': False}
        )

        
        recon_loss, kl_loss, contrastive_loss = self.module.loss(  
            tensors=batch,
            inference_outputs=inf_outputs,
            generative_outputs=gen_outputs,
            contrastive_loss_type=getattr(self, 'contrastive_loss_type', 'none'),
            supcon_temperature=getattr(self, 'supcon_temperature', 0.1),
            n_hard_negatives=getattr(self, 'n_hard_negatives', 5),
            n_hard_positives=getattr(self, 'n_hard_positives', 3),
            use_cell_type_contrast=getattr(self, 'use_cell_type_contrast', True),
            infonce_temperature=getattr(self, 'infonce_temperature', 0.15),
            multilevel_contrastive=getattr(self, 'multilevel_contrastive', True),
            pert_contrastive_weight=getattr(self, 'pert_contrastive_weight', 1.0),
            full_contrastive_weight=getattr(self, 'full_contrastive_weight', 0.0),
            false_negative_threshold=getattr(self, 'false_negative_threshold', 0.85),
        )

        if contrastive_loss is None:
            contrastive_loss = recon_loss.new_tensor(0.0)

        total_loss = recon_loss + self.kl_weight * kl_loss

        if getattr(self, 'contrastive_loss_type', 'none') != 'none':
            total_loss += self.contrastive_lambda * contrastive_loss

        self.log("val_loss", total_loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        if getattr(self, 'contrastive_loss_type', 'none') != 'none':
            self.log("val_contrastive", contrastive_loss)

        
        import math

        
        r2_mean, r2_var = self.module.r2_metric(batch, inf_outputs, gen_outputs, mode='direct')  
        disnt_basal, disnt_after = self.module.disentanglement(batch, inf_outputs, gen_outputs)  

        results = {}
        
        results.update({'r2_mean': r2_mean.item() if hasattr(r2_mean, 'item') else r2_mean,
                       'r2_var': r2_var.item() if hasattr(r2_var, 'item') else r2_var})
        results.update({'disnt_basal': disnt_basal.item() if hasattr(disnt_basal, 'item') else disnt_basal})
        results.update({'disnt_after': disnt_after.item() if hasattr(disnt_after, 'item') else disnt_after})
        results.update({'KL': kl_loss.item()})
        results.update({'recon_loss': recon_loss.item()})
        if getattr(self, 'contrastive_loss_type', 'none') != "none":
            results.update({'contrastive_loss': contrastive_loss.item()})
        else:
            results.update({'contrastive_loss': 0.0})

        
        
        disnt_alpha = 0.1
        disnt_diff = results['disnt_after'] - results['disnt_basal']
        if hasattr(self.module, 'recon_loss') and self.module.recon_loss in ['mse_sum', 'mse', 'huber']:
            scavae_metric = results['r2_mean'] + disnt_alpha * disnt_diff
        else:
            scavae_metric = results['r2_mean'] + 0.5 * results['r2_var'] + disnt_alpha * disnt_diff

        results.update({'scavae_metric': scavae_metric})

        return results

    def validation_epoch_end(self, outputs):
        
        import numpy as np

        
        r2_means = [out['r2_mean'] for out in outputs if 'r2_mean' in out]  
        r2_vars = [out['r2_var'] for out in outputs if 'r2_var' in out]  
        disnt_basals = [out['disnt_basal'] for out in outputs if 'disnt_basal' in out]  
        disnt_afters = [out['disnt_after'] for out in outputs if 'disnt_after' in out]  
        scavae_metrics = [out['scavae_metric'] for out in outputs if 'scavae_metric' in out]  
        recon_losses = [out['recon_loss'] for out in outputs if 'recon_loss' in out]  
        kl_losses = [out['KL'] for out in outputs if 'KL' in out]  
        contrastive_losses = [out['contrastive_loss'] for out in outputs if 'contrastive_loss' in out]  

        
        val_r2_mean = float(np.mean(r2_means)) if r2_means else 0.0
        val_r2_var = float(np.mean(r2_vars)) if r2_vars else 0.0
        val_disnt_basal = float(np.mean(disnt_basals)) if disnt_basals else 0.0
        val_disnt_after = float(np.mean(disnt_afters)) if disnt_afters else 0.0
        val_scavae_metric = float(np.mean(scavae_metrics)) if scavae_metrics else 0.0

        
        self.log("val_r2_mean", val_r2_mean, prog_bar=True)  
        self.log("val_r2_var", val_r2_var, prog_bar=False)
        self.log("disnt_basal", val_disnt_basal, prog_bar=True)
        self.log("disnt_after", val_disnt_after, prog_bar=True)
        self.log("scavae_metric", val_scavae_metric, prog_bar=True)

        if recon_losses:
            val_recon = float(np.mean(recon_losses))
            self.log("val_recon", val_recon, prog_bar=True)
        if kl_losses:
            val_kl = float(np.mean(kl_losses))
            self.log("val_kl", val_kl, prog_bar=True)
        if contrastive_losses and getattr(self, 'contrastive_loss_type', 'none') != 'none':
            val_contrastive = float(np.mean(contrastive_losses))
            self.log("val_contrastive", val_contrastive, prog_bar=True)

        
        self.epoch_history['epoch'].append(self.current_epoch)
        self.epoch_history['mode'].append('valid')

        
        self.epoch_history['train_loss'].append(0.0)
        self.epoch_history['recon_loss'].append(0.0)
        self.epoch_history['KL'].append(0.0)
        self.epoch_history['contrastive_loss'].append(0.0)

        
        self.epoch_history['r2_mean'].append(val_r2_mean)
        self.epoch_history['r2_var'].append(val_r2_var)
        self.epoch_history['disnt_basal'].append(val_disnt_basal)
        self.epoch_history['disnt_after'].append(val_disnt_after)
        self.epoch_history['scavae_metric'].append(val_scavae_metric)
