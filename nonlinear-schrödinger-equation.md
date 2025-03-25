# Enhanced PINN for a Coupled Gross–Pitaevskii Equation

This repository contains a Physics-Informed Neural Network (PINN) implementation that solves a coupled Gross–Pitaevskii (GP)-type (nonlinear Schrödinger) equation. The exact solution used is:

\[
\psi(x,t) = \operatorname{sech}(x) \, e^{i\,t}
\]

which is split into its real and imaginary parts:
- \( u(x,t) = \operatorname{sech}(x) \cos(t) \)
- \( v(x,t) = \operatorname{sech}(x) \sin(t) \)

## Overview

- **Physics-Informed Loss:**  
  The PINN enforces the governing equation by computing the PDE residuals using automatic differentiation. The residuals are derived by splitting the complex nonlinear Schrödinger equation into two real-valued equations:
  - \( -v_t + 0.5\, u_{xx} + (u^2+v^2)u = 0 \)
  - \( u_t + 0.5\, v_{xx} + (u^2+v^2)v = 0 \)

- **Fourier Features & Normalization:**  
  A Fourier feature layer is used to map the low-dimensional input \((x, t)\) to a higher-dimensional space to capture high-frequency variations. Input data is normalized to stabilize training.

- **Network Architecture:**  
  The enhanced model uses multiple fully connected layers with GELU activations and a final linear layer that outputs the two components \([u, v]\).

- **Adaptive Sampling & Training:**  
  Collocation points are generated adaptively, with extra sampling near the region where the solution is more sensitive. Training uses an AdamW optimizer with a OneCycleLR scheduler and early stopping based on improvement in total loss.

- **Visualization:**  
  The code provides several visualization modes including:
  - 2D plots at selected time slices.
  - Interactive slider plots to examine individual time snapshots.
  - 3D surface plots (using Plotly) for both the real and imaginary parts and the corresponding error heatmaps.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/madhushankara/your-repo-name.git
   ```
2. Open the notebook in Google Colab or run the Python script locally.
3. The main script will generate adaptive collocation points, train the PINN model, and produce the visualizations.
4. The trained model is saved as `schrodinger_pinn_model.pth` for future use.

## Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- Matplotlib  
- Plotly  
- ipywidgets  
- tqdm

