# Spectral Basis Graphs (SBG)

Official PyTorch repository for **"Spectral Basis Graphs: Compact Cross-Channel Modeling for Multivariate Time Series Forecasting"**.

## Overview

**Spectral Basis Graphs (SBG)** is a lightweight, plug-and-play residual adapter for multivariate time series (MTS) forecasting. Instead of materializing dense quadratic $\mathcal{O}(C^2)$ channel-to-channel graphs, SBG reformulates cross-variable interaction as sparse $\mathcal{O}(kC)$ bipartite routing from variates to a compact dictionary of $R$ orthogonal spectral bases ($R \ll C, k \le R$):

- **Stage (A) Dual-Domain Channel Encoding & Circular Phase Coherence**: Combines RevIN-standardized temporal tokens with top-$M$ rFFT harmonic coordinates and evaluates amplitude-weighted circular phase alignment on $\mathbb{S}^1$.
- **Stage (B) Differentiable Sparse Bipartite Routing via STE**: Enforces exact top-$k$ hard edge pruning in the forward pass to block localized sensor noise while preserving continuous gradient flow in the backward pass.
- **Stage (C) Self-State Subtraction & Validation-Constrained Calibration**: Deducts self-loop reconstructions ($\boldsymbol{\delta}_c = \mathbf{v}_c - \mathbf{h}_c^{\mathrm{self}}$) in $\mathcal{O}(kC)$ time and applies validation-calibrated horizon gating ($\boldsymbol{\alpha}^\star$) to guarantee safe residual fusion across host backbones.

## Repository Structure

```text
SBG/
├── configs/          # Benchmark & hyperparameter configurations (R=8, k=2 default)
├── layers/           # Dual-domain tokenizer, circular phase router (STE), bipartite adapter
├── models/           # Host wrappers (DLinear, PatchTST, iTransformer, TimeBase, MixLinear)
├── scripts/          # Reproduction scripts across 9 OpenTS / TFB benchmarks
└── requirements.txt  # Python & PyTorch dependencies
```

## Environment Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy, SciPy, scikit-learn, Pandas

## Benchmarks & Default Hyperparameters

Evaluated across 9 public benchmarks (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Exchange`, `Weather`, `Solar-Energy`, `Electricity`, `Traffic`) over horizons $H \in \{96, 192, 336, 720\}$:

| Regime | Basis Count $R$ | Routing Degree $k$ | Harmonics $M$ | Hidden Dim $d$ | Learning Rate $\eta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| High-dim ($C > 8$) | 8 | 2 | 16 | 512 | $1 \times 10^{-3}$ |
| Low-dim ($C \le 8$) | 4 | 1 | 16 | 32 | $5 \times 10^{-4}$ |

## Code Release Status

> **Note**: Core training, evaluation, and profiling scripts are currently being cleaned and organized for release and will be uploaded to this repository shortly.
