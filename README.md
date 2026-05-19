# scCAVAE

scCAVAE (Single-Cell Causal Autoencoder with Variational Attention and Embedding) is a deep generative model for predicting single-cell transcriptional responses to perturbations (drugs or genetic edits). It extends the CPA framework with enhanced perturbation encoding strategies and latent fusion mechanisms.

The model decomposes single-cell expression into three latent components: **basal latent** (encodes unperturbed cell state), **perturbation latent** (models drug effects via an optimized perturbation network), and **covariate latent** (captures batch effects, cell type, and other metadata). These are fused via addition, concatenation, or multi-head attention before being decoded to reconstruct gene expression.

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch 2.0, PyTorch Lightning 1.9.5, scvi-tools 0.20.3, AnnData, scanpy.

## Training

Training is invoked by calling `model.train()`, which uses scvi-tools' `TrainRunner` under the hood. In addition to standard PyTorch Lightning arguments (`max_epochs`, `batch_size`, `precision`, etc.), `plan_kwargs` configures the optimizer, adversarial training, contrastive loss, and MixUp schedule. See `train/` for complete per-dataset scripts.
