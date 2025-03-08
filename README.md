# PINN

The code implements a PINN for solving the harmonic oscillator ODE (
𝑢
¨
+
𝑢
=
0
u
¨
 +u=0) by combining the PDE residual loss with initial condition losses. It uses a fully connected network with Tanh activations to map time 
𝑡
t to the solution 
𝑢
(
𝑡
)
u(t). Automatic differentiation computes first and second derivatives to enforce the ODE. Adam optimizer with a learning rate scheduler refines the solution over many collocation points. Enhanced visualization includes 2D plots, interactive sliders, and 3D Plotly graphs for a comprehensive view of the exact versus predicted solutions and the error.
