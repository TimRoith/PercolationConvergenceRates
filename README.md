# Percolation Convergence Rates

<p float="left">
  <img src="https://user-images.githubusercontent.com/44805883/196187492-d142de95-f2bb-4db7-affa-0694da6aec97.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/44805883/196187868-4dff5e30-1e35-4373-9b2e-51f8c62aec2c.gif" width="400" />
</p>




Experimental evaluation of convergence rates for Euclidean first-passage percolation. This code produces the examples for the paper "Ratio convergence rates for Euclidean first-passage percolation: Applications to the graph infinity Laplacian" [1].

## What is computed 

We consider domains of the form
$
\begin{align*}
\Omega_s = [-s^{1/d}, s+s^{1/d}] \times \left[-s^{1/d}, s^{1/d}\right]^{d-1}
\end{align*}
$
for different values of $s>0$ and dimensions $d=2$ and $d=3$. 
In order to observe the limiting behavior for $s\to\infty$ we evaluate distances at $s_i = 100\cdot 2^i$ for $i=1,\ldots, N\in\N$. For each distance we perform $K\in\N$ different trials, where in each trial $k=1,\ldots,K$ we sample a Poisson point process $P_{i,k}\subset\Omega_{s_i}$ with unit density and then set

$
\begin{align*}
\overline{\mathrm{T}}_{i}:= \frac{1}{K}\sum_{k=1}^K d_{h_{s_i}, P_{i,k}}(0, s_i e_1).
\end{align*}
$

Here, the scaling is chosen as
%
\begin{align}\label{eq:logscaling}
h_s := a\log(s)^{1/d},
\end{align}
%
where $a>0$ is a factor.

### Convergence of $T_s/s$

We are interested in the values
$
\begin{align}\label{eq:sample_ratio}
\frac{\overline{\mathrm{T}}_i}{s_i},\quad i=1,\ldots, N.
\end{align}
$

The figure on the right is a log-log plot of the values
$
\begin{align*}
\abs{\frac{\overline{\mathrm{T}}_i}{s_i} - \sigma},\qquad i=1,\dots,N-1,
\end{align*}
$
is visualized, where the limiting constant $\sigma$, whose analytic value is not known, is approximated by $\sigma = \frac{\overline{\mathrm{T}}_N}{s_N}$.

### Ratio Convergence

Additionally, we want to evaluate the ratio convergence which we proved in \cref{prop:ratio_convergence_exp}. 
Therefore, we also compute

$
\begin{align*}
\overline{\mathrm{T}}_{i,1/2}:= \frac{1}{K}\sum_{k=1}^K d_{h_{s_i}, P_{i,k}}\left(0, \frac{s_i}{2} e_1\right)
\end{align*}
$

and visualize the ratios and errors

$
\begin{align*}
\frac{\overline{\mathrm{T}}_{i}}{\overline{\mathrm{T}}_{i,1/2}},
\qquad
\abs{\frac{\overline{\mathrm{T}}_{i}}{\overline{\mathrm{T}}_{i,1/2}} - \frac{1}{2}}
\end{align*}
$

for $i=1,\ldots, N$.
