[Table of Contents](http://nbviewer.ipython.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb)

<pre data-code-language="python"
     data-executable="true"
     data-type="programlisting">
#format the book
%matplotlib inline
from __future__ import division, print_function
import matplotlib.pyplot as plt
import book_format
book_format.load_style()
</pre>

# Symbology

This is just notes at this point.

## State

<span class="math-tex" data-type="tex">\\(x\\)</span> (Brookner, Zarchan, Brown)

<span class="math-tex" data-type="tex">\\(\underline{x}\\)</span> Gelb)

## State at step n

<span class="math-tex" data-type="tex">\\(x_n\\)</span> (Brookner)

<span class="math-tex" data-type="tex">\\(x_k\\)</span> (Brown, Zarchan)

<span class="math-tex" data-type="tex">\\(\underline{x}_k\\)</span> (Gelb)



## Prediction

<span class="math-tex" data-type="tex">\\(x^-\\)</span>

<span class="math-tex" data-type="tex">\\(x_{n,n-1}\\)</span>  (Brookner)

<span class="math-tex" data-type="tex">\\(x_{k+1,k}\\)</span>


## measurement


<span class="math-tex" data-type="tex">\\(x^*\\)</span>



Y_n (Brookner)

##control transition Matrix

<span class="math-tex" data-type="tex">\\(G\\)</span> (Zarchan)


Not used (Brookner)

#Nomenclature


## Equations
### Brookner

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
X^*_{n+1,n} &= \Phi X^*_{n,n} \\
X^*_{n,n}  &= X^*_{n,n-1} +H_n(Y_n - MX^*_{n,n-1}) \\
H_n &= S^*_{n,n-1}M^T[R_n + MS^*_{n,n-1}M^T]^{-1} \\
S^*_{n,n-1} &= \Phi S^*_{n-1,n-1}\Phi^T + Q_n \\
S^*_{n-1,n-1} &= (I-H_{n-1}M)S^*_{n-1,n-2}
\end{aligned}\\)</span>

### Gelb

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\underline{\hat{x}}_k(-) &= \Phi_{k-1} \underline{\hat{x}}_{k-1}(+) \\
\underline{\hat{x}}_k(+)  &= \underline{\hat{x}}_k(-) +K_k[Z_k - H_k\underline{\hat{x}}_k(-)] \\
K_k &= P_k(-)H_k^T[H_kP_k(-)H_k^T + R_k]^{-1}\\
P_k(+) &=  \Phi_{k-1} P_{k-1}(+)\Phi_{k-1}^T + Q_{k-1} \\
P_k(-) &= (I-K_kH_k)P_k(-)
\end{aligned}\\)</span>


### Brown

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\hat{\textbf{x}}^-_{k+1} &= \mathbf{\phi}_{k}\hat{\textbf{x}}_{k} \\
\hat{\textbf{x}}_k  &= \hat{\textbf{x}}^-_k +\textbf{K}_k[\textbf{z}_k - \textbf{H}_k\hat{\textbf{}x}^-_k] \\
\textbf{K}_k &= \textbf{P}^-_k\textbf{H}_k^T[\textbf{H}_k\textbf{P}^-_k\textbf{H}_k^T + \textbf{R}_k]^{-1}\\
\textbf{P}^-_{k+1} &=  \mathbf{\phi}_k \textbf{P}_k\mathbf{\phi}_k^T + \textbf{Q}_{k} \\
\mathbf{P}_k &= (\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf{P}^-_k
\end{aligned}\\)</span>


### Zarchan

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\hat{x}_{k} &= \Phi_{k}\hat{x}_{k-1} + G_ku_{k-1} + K_k[z_k - H\Phi_{k}\hat{x}_{k-1} - HG_ku_{k-1} ] \\
M_{k} &=  \Phi_k P_{k-1}\phi_k^T + Q_{k} \\
K_k &= M_kH^T[HM_kH^T + R_k]^{-1}\\
P_k &= (I-K_kH)M_k
\end{aligned}\\)</span>

### Wikipedia
<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\hat{\textbf{x}}_{k\mid k-1} &= \textbf{F}_{k}\hat{\textbf{x}}_{k-1\mid k-1} + \textbf{B}_{k} \textbf{u}_{k} \\
\textbf{P}_{k\mid k-1} &=  \textbf{F}_{k} \textbf{P}_{k-1\mid k-1} \textbf{F}_{k}^{\text{T}} + \textbf{Q}_{k}\\
\tilde{\textbf{y}}_k &= \textbf{z}_k - \textbf{H}_k\hat{\textbf{x}}_{k\mid k-1} \\
\textbf{S}_k &= \textbf{H}_k \textbf{P}_{k\mid k-1} \textbf{H}_k^\text{T} + \textbf{R}_k \\
\textbf{K}_k &= \textbf{P}_{k\mid k-1}\textbf{H}_k^\text{T}\textbf{S}_k^{-1} \\
\hat{\textbf{x}}_{k\mid k} &= \hat{\textbf{x}}_{k\mid k-1} + \textbf{K}_k\tilde{\textbf{y}}_k \\
\textbf{P}_{k|k} &= (I - \textbf{K}_k \textbf{H}_k) \textbf{P}_{k|k-1}
\end{aligned}\\)</span>

### Labbe

<span class="math-tex" data-type="tex">\\(
\begin{aligned}
\hat{\textbf{x}}^-_{k+1} &= \mathbf{F}_{k}\hat{\textbf{x}}_{k} + \mathbf{B}_k\mathbf{u}_k \\
\textbf{P}^-_{k+1} &=  \mathbf{F}_k \textbf{P}_k\mathbf{F}_k^T + \textbf{Q}_{k} \\
\textbf{y}_k &= \textbf{z}_k - \textbf{H}_k\hat{\textbf{}x}^-_k \\
\mathbf{S}_k &= \textbf{H}_k\textbf{P}^-_k\textbf{H}_k^T + \textbf{R}_k \\
\textbf{K}_k &= \textbf{P}^-_k\textbf{H}_k^T\mathbf{S}_k^{-1} \\
\hat{\textbf{x}}_k  &= \hat{\textbf{x}}^-_k +\textbf{K}_k\textbf{y} \\
\mathbf{P}_k &= (\mathbf{I}-\mathbf{K}_k\mathbf{H}_k)\mathbf{P}^-_k
\end{aligned}\\)</span>
