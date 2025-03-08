# PINN

The code implements a PINN for solving the harmonic oscillator ODE (ü + u = 0) by combining the PDE residual loss with initial condition losses. It uses a fully connected network with Tanh activations to map time t to the solution u(t). Automatic differentiation computes first and second derivatives to enforce the ODE. Adam optimizer with a learning rate scheduler refines the solution over many collocation points. Enhanced visualization includes 2D plots, interactive sliders, and 3D Plotly graphs for a comprehensive view of the exact versus predicted solutions and the error.
