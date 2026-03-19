# Project Context: Modelling the Olfactory Bulb (OB)

**Technical Milestone Report context** — Use this file as reference when working on this codebase.

## Summary

Mathematical model of the **mouse olfactory bulb** for investigating **homeostatic plasticity**. The model includes all major neuron types and their main interactions. Key findings: substantial **parameter degeneracy** (many configurations produce the same output), **stability** to small perturbations, and **Hessian-based** directions in parameter space that preserve output—relevant to experimentally observed homeostatic compensation. Implemented with **JAX/Optax**; future work focuses on constraining solutions to biologically plausible synaptic changes.

---

## Biological & Experimental Background

- **Sensory deprivation**: Blocking one nostril (nose plug) affects only the ipsilateral OB. 24-hour deprivation causes **dopaminergic axonic neurons to shrink** in axonal length (Galliano et al.). After plug removal, **OB output returns to original level** despite structural changes—suggesting compensatory mechanisms elsewhere in the OB.
- **Data**: Experimental data from the **Galliano Laboratory**; OB is compact and well-characterised.

---

## Neuron Types (by colour/alias)

| Alias | Type | Role |
|-------|------|------|
| **b** (blue) | Anaxonic interneurons | Self-inhibition within a glomerulus |
| **g** (green) | Axonic interneurons | Long-range lateral inhibition (between glomeruli) |
| **r** (red) | External tufted cells (ETCs) | Relay glomeruli → mitral cells; sister glomeruli merged in model |
| **y** | Mitral cells | Integrate from glomeruli; **primary output to cortex** |
| **gr** | Granule cells | Feedback inhibition to mitral cells; receive cortical input (often excluded in initial analysis) |

- **OSNs** (olfactory sensory neurons): input; same receptor type → same glomerulus.
- Model uses **two glomeruli** to capture cross-glomerular inhibition (green); blue inhibits only own glomerulus.

---

## Mathematical Model

### Neuron / transfer function

- **Naka–Rushton** (steady-state firing rates, differentiable):
  - \( \sigma_j(x) = \frac{x^{n_j}}{x^{n_j} + k_j^{n_j}} \). For efficiency, inputs scaled by \(1/k_j\) and scaling absorbed into weights: \( \sigma_j(x') = \frac{x'^{n_j}}{x'^{n_j} + 1} \).
  - \(k_j\): semi-saturation constant; \(n_j\): Hill coefficient. Report uses **n = 4**.
- Weights \(w_{i,j}\) are synaptic strengths (and can absorb \(1/k_j\)).

### OB steady-state equations

Firing rates: **b** (blue), **r** (red), **g** (green), **y** (mitral), **gr** (granule). \(\alpha \in [0,1]\): inhibitory feedback from blue to OSNs (OSN output = fraction \(\alpha\) of original). \(W_{i,j}\) are **weight matrices** of form `[0, w_i1,j1; w_i2,j2, 0]` (cross-glomerular).

```
b = σ(w_osn,b · α · x)
r = σ(w_osn,r · α · x − w_b,r·b − W_g,r·g)
g = σ(w_osn,g · α · x − w_b,g·b + w_r,g·r)
y = σ(w_osn,y · α · x − w_b,y·b − W_g,y·g + w_r,y·r − w_gr,y·gr)
gr = σ(w_cor,gr·cortex + w_y,gr·y)
```

- **Fixed-point 1**: Eqs. for **g, r** (coupled) → solved with **Anderson acceleration** (history 5, tolerance 1e-6, max 50 iterations).
- **Fixed-point 2**: **y, gr** (granule); granule cells often **excluded** in initial analysis (cortex poorly characterised, homeostasis assumed local).

---

## Parameter Set θ and Fitting

- **θ = {ω, α}**: weights ω and feedback α.
- **Target**: output **y** and firing rates **r, g, b, y** near **0.5** (maximise dynamic range).
- **Loss** (over mitral cells \(i=1..m\), \(m=2\)):
  - \( L(\theta) = \sum_i \left[ (r_i - r_{t,i})^2 + (g_i - g_{t,i})^2 + (b_i - b_{t,i})^2 + (y_i - y_{t,i})^2 \right] \), with targets \(r_{t,i}, g_{t,i}, b_{t,i}, y_{t,i} = 0.5\).
- **Optimiser**: **Adam** (β1=0.9, β2=0.999), **JAX/Optax**, JIT.
- **Constraints**: parameters > 0; α ∈ [0,1]; **diagonal of weight matrices W = 0**.
- **Degeneracy**: many θ give same target output and firing rates.

---

## Stability Analysis

- **State** \(s(t) = [r(t), g(t), b(t), y(t)]\); steady state \(s^*\): \(F(s^*, \theta, x) = s^*\).
- Linearisation: \(e(t+1) \approx J\, e(t)\), \(J = \partial F/\partial s\big|_{s^*}\). Continuous form: \(\frac{de}{dt} = \frac{1}{\tau}(J - I)e = A e\).
- **Stability**: fixed point **locally asymptotically stable** iff all eigenvalues of **A** have **negative real parts** (τ only scales, doesn’t change sign). Report: 1000 configurations all had eigenvalues with negative real parts (many near −1 from weak coupling / zero diagonal in J).

---

## Homeostatic Plasticity: Δθ Preserving Output

- **Goal**: find **Δθ** such that \(y(\theta + \Delta\theta) = y(\theta)\) (output-preserving changes).
- Taylor: \(y(\theta_o + \Delta\theta) \approx y(\theta_o) + g^T\Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta\). With **g** = ∇_θ y, **H** = ∇²_θ y (Hessian).
- **Condition**: \(g^T\Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta \approx 0\).
- **Focus**: Δθ **perpendicular to g** and along **eigenvectors of H with near-zero eigenvalues** → family of compensatory directions (homeostatic plasticity).
- **Eigenvalues of H**: \(|\lambda_i| \approx 0\) (e.g. < 1e-6) → directions in parameter space where synaptic weights can change without much effect on OB output.
- **Eigenvectors** of those near-zero eigenvalues (and ⊥ to g) show **which parameters** (synaptic w, α, etc.) can co-vary.
- **Participation ratio (PPR)** of Hessian eigenvalues:  
  \( \mathrm{PPR} = \frac{\big(\sum_i |\lambda_i|\big)^2}{\sum_i \lambda_i^2} \).  
  **Smaller PPR** → fewer dominant directions → **more** directions that preserve output. Can filter parameter sets by PPR.

---

## Implementation Notes

- **JAX** (and **Optax**) for optimisation and automatic differentiation; JIT where used.
- **Anderson acceleration** for fixed-point iteration (g, r).
- Parameters: **positive**; **α ∈ [0,1]**; **diagonal of W = 0**.
- Granule cells (gr) and cortex input often **excluded** in initial analyses.

---

## Next Steps (from report)

- Constrain parameter space: **PPR-based filtering**, excitatory–inhibitory balance, biologically plausible Naka–Rushton parameters, exclude silencing/saturating eigen-directions, consistency with **dopaminergic axon shrinkage**.
- If systematic synaptic patterns emerge, relate to OB biology and inform experiments.

---

## References (key)

- Galliano et al., J Neurosci, 41, 2135, 2021 (sensory deprivation, axon shrinkage)
- Galliano Lab: https://gallianolab.org/
- Nagayama et al., Front Neural Circuits, 8, 98944, 2014 (OB circuitry)
- Walker et al., Anderson acceleration, SIAM J Numer Anal, 49, 1715, 2011
- Kingma & Ba, Adam, ICLR 2015
- JAX: github.com/jax-ml/jax
