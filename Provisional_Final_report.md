# Provisional report: mitral-circuit parameterisation, stability, and null-space structure

This draft mirrors the section order in `TTFinal_report.ipynb` and summarises the modelling steps in plain language so the notebook and the written report stay aligned.

---

## 1. Mathematical model and Naka–Rushton nonlinearity

The reduced olfactory-bulb style circuit maps OSN input \(x \in \mathbb{R}^m\) (here \(m=2\) glomeruli) to mitral and interneuron firing rates through a **Naka–Rushton** static nonlinearity,

\[
f(x; n) = \frac{x^n}{x^n + 1},
\]

applied elementwise after ensuring non-negative inputs (so fractional Hill exponents remain well-defined in code). Different cell classes (red / blue / green / yellow populations in your notation) can carry **different** Hill coefficients \(n_r, n_b, n_g, n_y\). Interneuron activity feeds subtractively into mitral inputs; green mitral variables are defined implicitly as the **fixed point** of a coupled map solved with Anderson acceleration.

---

## 2. Symmetric weights (half the parameter count)

Instead of independent weights for each glomerulus, **symmetry** was imposed: each scalar weight is shared across the two mitral channels. Recurrent blocks \(W_{g\to r}\) and \(W_{g\to y}\) were structured as symmetric off-diagonal \(2\times2\) matrices controlled by one scalar each. The optimisation vector therefore has **12** learnable scalars per solution, rather than the full asymmetric layout.

---

## 3. Fixing \(n\) using ELISA / ephys curves

Hill exponents were **not** treated as free parameters in the main 12-dimensional sweep. They were estimated separately by fitting Naka–Rushton curves (with an additional scale / offset where needed) to **published ephys data** (Galliano et al., Excel sheets for mitral, red, green, blue populations under control vs occlusion). The fitted values (or rounded literature defaults such as \(n_r=2.333\), \(n_b=0.9\), \(n_g=2.0626\), \(n_y=1.562\)) are then **fixed** when optimising weights and when analysing stability and curvature.

---

## 4. Random initialisation and optimisation to a target output

Many initial weight vectors \(\theta^{(k)}\) were drawn at random. For each draw, **Adam** minimised a sum of squared errors between model outputs \((r,g,b,y)\) at the nominal input and **preset target firing rates** (your notebook used strong drive \(x=\mathbf{1}\) and targets such as \(0.5\) on selected channels). Only runs whose loss fell below a tolerance were kept. In the archived ensemble used later, **626 out of 1000** random starts converged to valid solutions; the failed runs were discarded.

---

## 5. Linear stability of the converged operating point

For each accepted \(\theta^\*\), the code forms the **discrete-time** state map \(F\) over \(s=[r,g,y]\), evaluates its Jacobian \(J\) at the fixed point \(s^\*\), builds \(A=(J-I)/\tau\), and checks that **all eigenvalues of \(A\)** have negative real parts (linearly stable dynamics in the chosen time-discretisation picture). For the **626** archived samples, **every** converged parameter set was classified as **stable**—no unstable survivors remained after filtering.

---

## 6. Output Hessian and eigenvalue spectrum

To study **sloppy** directions (combinations of parameters that barely move the readout), the scalar objective \(\sum_i y_i\) was differentiated twice in JAX to obtain the **Hessian** \(H\) at each \(\theta^\*\). Eigenvalues of \(H\) were sorted by magnitude. Across samples, the spectrum showed a clear gap: **four eigenvalues were systematically much smaller** than the rest, indicating a four-dimensional **near-null** subspace of the output map at convergence.

---

## 7. Concatenation, SVD, and sparse rotations

For each sample, eigenvectors whose eigenvalues fell below a small threshold were collected. **All** such vectors from **all** samples were **stacked** and submitted to an **SVD**. The leading right singular vectors (equivalently, the dominant rows of \(V^\top\) after the usual convention) summarise directions that recur across the ensemble. Geometrically, you restricted attention to the span of the **first four** such directions.

Within that four-dimensional subspace you sought **interpretable** axes: (i) a **varimax** rotation, and (ii) an **orthogonal** rotation found by minimising an **L1** proxy for sparsity (mean absolute loadings) using Adam over \(Q \in \mathrm{O}(4)\). The two procedures need not agree; comparing them motivated a more exhaustive search.

---

## 8. Multiple L1 minima: 10 000 initial \(Q\), histogram, and nested peaks

Because the L1 rotation problem is **non-convex**, `sparse_rotation` was re-run from **many random orthogonal initialisations** (order \(10^4\)). The terminal L1 losses formed a **histogram with three prominent peaks**. Zooming into the **lowest-loss** peak revealed **sub-structure**: effectively **two narrower peaks**. Analysing the corresponding rotation matrices showed that solutions in each sub-peak are related by **signed permutations** of a small set of columns—interpretable as **two “families”** of sparse bases, each family being combinations of **two sets of four** fundamental sparse patterns (in the sense of column permutations / sign flips within the four-dimensional subspace).

---

## 9. Optional: clustering on the broad low-loss cluster

The same pairwise-distance / clustering machinery can be applied to the **entire** broad low-loss mode in the L1 histogram, not only the tightest sub-peak. This checks how many distinct families appear when more solutions are included.

---

## 10. Stepping in \(\Delta\theta\) along sparse directions

Using a representative sparse basis \(U_{\mathrm{sparse}}\) (and sometimes augmenting with a random fifth direction), parameters were perturbed as \(\theta(\varepsilon)=\theta^\*+\varepsilon\, u_j\). The notebook plots **relative changes** in mitral outputs \(y\) as a function of scaled step size, illustrating which directions move the output quickly vs leave it flat—connecting the abstract Hessian null space to **operational** sensitivity.

---

## 11. Extension: constrained directions and learnable \(n_g\)

A later experiment **augmented** \(\theta\) with a **learnable green Hill exponent** \(n_g\) as a **13th** parameter (still with symmetric weights elsewhere), re-optimised under a different loss pattern (e.g. emphasis on \(y\) at two input strengths), and stored a smaller archive (e.g. **117** samples). For those solutions, the Hessian analysis was repeated **after projecting out** a fixed constraint direction in parameter space (orthogonal projector applied to \(H\)), and **SVD / heatmaps** were used to isolate a special “\(\mu^\*\)” direction visible in the constrained Jacobian–Hessian pipeline. This section is logically **after** the main 12-parameter story because it **redefines** `unpack_theta` and `mitral_output`.

---

## Closing remark

Together, the pipeline is: **biophysically shaped nonlinearities** \(\to\) **symmetric weight reduction** \(\to\) **data-informed \(n\)** \(\to\) **multi-start fitting** \(\to\) **linear stability certification** \(\to\) **curvature / null-space geometry** \(\to\) **sparse basis discovery with multi-modal L1 landscape** \(\to\) **optional broad-cluster analysis** \(\to\) **local sensitivity plots along sparse directions** \(\to\) **constrained extension with learnable \(n_g\)**.

The collated notebook `TTFinal_report.ipynb` reproduces the corresponding code blocks in **eleven** numbered sections (§1–§11), including the optional broad-cluster analysis as §9.
