# :coffee: Percolation Convergence Rates

<p float="left">
  <img src="https://user-images.githubusercontent.com/44805883/196187492-d142de95-f2bb-4db7-affa-0694da6aec97.gif" width="400" />
  <img src="https://user-images.githubusercontent.com/44805883/196187868-4dff5e30-1e35-4373-9b2e-51f8c62aec2c.gif" width="400" />
</p>




Experimental evaluation of convergence rates for Euclidean first-passage percolation. This code produces the examples for the paper "Ratio convergence rates for Euclidean first-passage percolation: Applications to the graph infinity Laplacian".

```
@article{bungert2022ratio,
    author = {Bungert, Leon and Calder, Jeff and Roith, Tim},
    title = {Ratio convergence rates for Euclidean first-passage percolation: Applications to the graph infinity Laplacian},
    year = {2022}
}
```

## :bulb: What is computed 

We consider domains of the form

$$
\begin{align}
\Omega_s = [-s^{1/d}, s+s^{1/d}] \times \left[-s^{1/d}, s^{1/d}\right]^{d-1}
\end{align}
$$

for different values of $s>0$ and dimensions $d=2$ and $d=3$. 
In order to observe the limiting behavior for $s\to\infty$ we evaluate distances at $s_i = 100\cdot 2^i$ for $i=1,\ldots, N\in\mathbb{N}$. For each distance we perform $K\in\mathbb{N}$ different trials, where in each trial $k=1,\ldots,K$ we sample a Poisson point process $P_{i,k}\subset\Omega_{s_i}$ with unit density and then set

$$
\begin{align*}
\overline{\mathrm{T}}_{i}:= \frac{1}{K}\sum_{k=1}^K d_{h_{s_i}, P_{i,k}}(0, s_i e_1).
\end{align*}
$$

Here, the scaling is chosen as

$$
\begin{align}
h_s := a\log(s)^{1/d},
\end{align}
$$

where $a>0$ is a factor. The file ```compute_dist.py``` computes the distances $d_{h_{s_i}, P_{i,k}}(0, s_i e_1)$ for each trial and each value of $s_i$ and writes them in to a dedicated .csv file. Evaluation is then performed in the script ```convergence_rate.py``` in the subfolder ```/results```.

## :chart_with_upwards_trend: Convergence of $T_s/s$

|<img alt="image" src="https://user-images.githubusercontent.com/44805883/196193101-dd7f0b2d-fba4-41f2-bd4b-44abf6821aa1.png"  width="498">|<img alt="image" src="https://user-images.githubusercontent.com/44805883/196192680-7e0df67e-5b2b-4c85-83ef-413ede299deb.png"  width="498" align="center">|
|:--:|:--:| 
|Dimension $d=2$|Dimension $d=3$|

We are interested in the values
  
$$
\begin{align}
\frac{\overline{\mathrm{T}}_i}{s_i},\quad i=1,\ldots, N.
\end{align}
$$

The right side is a log-log plot of the values

$$
\begin{align*}
\left|\frac{\overline{\mathrm{T}}_i}{s_i} - \sigma\right|,\qquad i=1,\dots,N-1,
\end{align*}
$$

is visualized, where the limiting constant $\sigma$, whose analytic value is not known, is approximated by $\sigma = \frac{\overline{\mathrm{T}}_N}{s_N}$.

## :chart_with_upwards_trend: Ratio Convergence

|<img width="500" alt="image" src="https://user-images.githubusercontent.com/44805883/196195548-1b794453-a213-42bd-a23b-12bb533cfaf2.png">|<img width="501" alt="image" src="https://user-images.githubusercontent.com/44805883/196195774-18ff914b-04cd-4f8b-90ae-623fe44d3313.png">|
|:--:|:--:| 
|Dimension $d=2$|Dimension $d=3$|

Additionally, we want to evaluate the ratio convergence which we proved in \cref{prop:ratio_convergence_exp}. 
Therefore, we also compute

$$
\begin{align*}
\overline{\mathrm{T}}_{i,1/2}:= \frac{1}{K}\sum_{k=1}^K d_{h_{s_i}, P_{i,k}}\left(0, \frac{s_i}{2} e_1\right)
\end{align*}
$$

and visualize the ratios and errors

$$
\begin{align*}
\frac{\overline{\mathrm{T}}_{i}}{\overline{\mathrm{T}}_{i,1/2}},
\qquad
\left|\frac{\overline{\mathrm{T}}_{i}}{\overline{\mathrm{T}}_{i,1/2}} - \frac{1}{2}\right|
\end{align*}
$$

for $i=1,\ldots, N$.
