# PINN (Physics-Informed Neural Network)

The code implements a PINN for solving the harmonic oscillator ODE (ü + u = 0) by combining the PDE residual loss with initial condition losses. It uses a fully connected network with Tanh activations to map time t to the solution u(t). Automatic differentiation computes first and second derivatives to enforce the ODE. Adam optimizer with a learning rate scheduler refines the solution over many collocation points. Enhanced visualization includes 2D plots, interactive sliders, and 3D Plotly graphs for a comprehensive view of the exact versus predicted solutions and the error.

# PINN for a Gross–Pitaevskii (GP)-type (nonlinear Schrödinger) equation

This repository contains a Physics-Informed Neural Network (PINN) implementation that solves a coupled Gross–Pitaevskii (GP)-type (nonlinear Schrödinger) equation. The exact solution used is:

$$
\psi(x,t) = {sech}(x) \, e^{i\,t}
$$

which is split into its real and imaginary parts:

- $u(x,t) = {sech}(x)\cos(t)$  
- $v(x,t) = {sech}(x)\sin(t)$

## Overview

- **Physics-Informed Loss:**  
  The PINN enforces the governing equation by computing the PDE residuals using automatic differentiation. The residuals are derived by splitting the complex nonlinear Schrödinger equation into two real-valued equations:

- $-v_t + 0.5\, u_{xx} + (u^2+v^2)u = 0$  
- $u_t + 0.5\, v_{xx} + (u^2+v^2)v = 0$

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
   git clone https://github.com/madhushankara/your-repo-name.git](https://github.com/madhushankara/PINN/blob/1d64fdeca40c963d29e6a434fb8febbfe3bd36d2/schrodinger_equation.ipynb
   ```
2. Open the notebook in Google Colab or run the Python script locally.
3. The main script will generate adaptive collocation points, train the PINN model, and produce the visualizations.
4. The trained model is saved as `schrodinger_pinn_model.pth` for future use.

## Requirements

- Python 3
- PyTorch  
- NumPy  
- Matplotlib  
- Plotly  
- ipywidgets  
- tqdm

