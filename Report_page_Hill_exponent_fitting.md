# Hill-exponent calibration from published ephys (complement to `TTFinal_report_1_n_fitting.ipynb`)

This page documents **what the notebook does**, **which data it uses**, and **how the fitted (or default) Hill exponents connect to the rest of the project**. It is meant to be read alongside the executable notebook, which contains the plots and numerical printouts.

---

## Role in the wider report

The main modelling pipeline (symmetric weights, multi-start optimisation, linear stability, Hessian null-space analysis, sparse rotations, and optional extensions such as learnable \(n_g\)) treats the static input–output nonlinearity as **fixed** once the Hill exponents are chosen. The notebook `TTFinal_report_1_n_fitting.ipynb` is the **empirical bridge** between published firing-rate curves and those fixed values \((n_r, n_b, n_g, n_y)\). Anything that depends on `mitral_output` in the stability or curvature notebooks should be consistent with the assignments recorded here.

---

## 1. Nonlinearity used in code and in the model

The activation law is the **Naka–Rushton** (Hill) saturating function, applied elementwise after non-negative clipping so fractional exponents remain real:

\[
f(x; n) = \frac{x^n}{x^n + 1}, \qquad x \leftarrow \max(x, 0).
\]

In the notebook, \(f\) is implemented in JAX as `nr(x, n)` using `jnp.maximum` and powers of \(x\). A short cell overlays **four** curves for \(n \in \{0.9,\, 1.562,\, 2.0626,\, 2.333\}\), which are the values ultimately used when wiring parameters into downstream code. That figure is a sanity check: it shows how sensitive the saturating shape is to \(n\) before any curve fitting.

---

## 2. Software stack and global modelling constants

The opening code cell imports **JAX / JAX NumPy** (for `nr` and for consistency with the rest of the project), **NumPy**, **Matplotlib**, **SciPy** (`curve_fit`, `stats`), **pandas** (Excel ingestion), and several other libraries also used elsewhere in the report notebooks (`jaxopt`, `seaborn`, `sklearn`, etc.). It also sets `m = 2` glomeruli and `x = jnp.ones(m)` as placeholders aligned with the symmetric two-channel setup in the main model. Those lines are not central to the Excel fitting itself but keep the notebook in the same “two OSN inputs” convention as the optimisation notebooks.

---

## 3. Data sources and experimental contrasts

All curves come from **Galliano et al. (2021)** supplementary Excel material, stored under `elisa_nr_data/`. The notebook systematically loads **two conditions** where available—typically **unperturbed vs occluded** (or analogous column blocks)—so you can inspect raw traces before averaging.

| Population in the model | Workbook | Sheet | Notebook action |
|---------------------------|----------|-------|-------------------|
| **Mitral** | `Galliano et al 2021 - fig 4 ephys.xlsx` | `repetitive firing` | Reads unperturbed columns `A,G:AB` and occluded `A,AG:BE` (after `skiprows=3`), `nrows=45`. Plots both dataframes, then fits on the **unperturbed** block. |
| **Red (ETCs)** | `Galliano et al 2021 - fig 5 ephys.xlsx` | `ETC repetitive firing` | Unperturbed: `J,N:AQ` after `skiprows=4`, `nrows=35`, `dropna`, cast to float. Occluded: same columns with `skiprows=41`, `nrows=73`. Fits use **unperturbed** rows with the **last five stimulus levels dropped** (`iloc[:-5]`) to avoid edge behaviour at the highest inputs. |
| **Green (DA / model “green”)** | `Galliano et al 2021 - DA cells ephys repetitive firing.xlsx` | `DA cells (Fig 7 and 8)` | Unperturbed: `A,C:L`; occluded: `A,Y:AH`. Two columns (`B.4`, `D.1`) are **dropped** from the green unperturbed frame before averaging and fitting. |
| **Blue** | Same DA workbook and sheet | Unperturbed: `A,N:V`; occluded: `A,AJ:AP`. | Same plotting pattern; the fitting section includes two variant `nr_full` definitions (see below). |

**Yellow** interneurons: this notebook **does not** load a separate yellow curve. The final cell assigns `n_y = 1.562` from the **same literature-aligned default set** used in the stability / Hessian pipeline, to be pasted into `mitral_output` if you are not overriding it from elsewhere.

---

## 4. Fitting procedure (common structure)

For each cell class the workflow is:

1. **Index column** of the spreadsheet becomes the stimulus axis \(x\) (cast to `float`).
2. **Across-animal columns** are averaged row-wise to obtain a mean response curve \(y(x)\).
3. A **composite curve** `nr_full` is defined. It always rescales the stimulus as \(x/b\) before applying `nr(·, n)`, then multiplies by an **amplitude** chosen so the model can match the vertical scale of the data (either a separate `y_max` parameter fitted by `curve_fit`, or `y.max()` times an optional scalar factor).
4. **`scipy.optimize.curve_fit`** finds parameters. Initial guesses are typically `p0=[3, 200]` for \((n, b)\) or `p0=[3, 200, 100]` when `y_max` is also free. Mitral uses the default `curve_fit` settings; blue’s three-parameter variant sets `maxfev=8000` for robustness.

### 4.1 Mitral (fig. 4 ephys)

- `nr_full(x, n, b, y_max) = nr(x/b, n) * y_max` — **three** free parameters.
- Fitted on **unperturbed** mitral data only (printed `n` and `b`). How that numeric \(n\) is mapped onto the symbols \((n_r,n_b,n_g,n_y)\) in `mitral_output` follows your circuit labelling in the optimisation notebooks; the closing assignment cell lists the four values actually used when freezing the nonlinearity for the main 12-parameter runs.

### 4.2 Red ETCs (`n_r`)

- `nr_full(x, n, b) = nr(x/b, n) * y.max()` — **two** free parameters; amplitude fixed from the data’s peak.
- Rows trimmed with `iloc[:-5]` as noted above.

### 4.3 Green (`n_g`)

- Same two-parameter pattern as red, but with an extra **empirical vertical factor 1.3** on the NR output: `nr(x/b, n) * y.max() * 1.3`. That factor is a deliberate vertical stretch of the template curve relative to the data scale (documented here so it is not mistaken for a fitted biological parameter).

### 4.4 Blue (`n_b`)

- The notebook contains **two consecutive fits** on the blue unperturbed mean curve: first a **three-parameter** `nr_full(x, n, b, y_max)` (with commented-out two-parameter code nearby), then a **two-parameter** variant with `nr(x/b, n) * y.max() * 1.3` analogous to green. The printed parameters from the cell you treat as authoritative should match the version you adopt in the written report; the final hard-coded `n_b = 0.9` in the closing cell is the value carried into the downstream notebooks in this project snapshot.

---

## 5. Export block: assigning `n_r`, `n_b`, `n_g`, `n_y`

The last markdown cell instructs you to **copy printed fitted values** from the cells above **or** fall back to the **pre-agreed defaults** used in the stability and Hessian notebooks. The closing code cell fixes:

```text
n_r = 2.333
n_b = 0.9
n_g = 2.0626
n_y = 1.562
```

These are the numbers that should appear wherever the full four-population nonlinearity is frozen before weight optimisation.

---

## 6. Implementation notes and reproducibility

- **`nr` in `curve_fit`:** SciPy differentiates through the Python function; using JAX arrays inside `nr` is acceptable here because the optimiser drives scalar NumPy arrays through the call path in practice. If you refactor, keep `nr` numerically identical to the JAX version used in training.
- **Occluded panels:** They are loaded and plotted for **visual comparison** with unperturbed curves; the **fits** in this notebook target **unperturbed** (or trimmed unperturbed) means unless you explicitly change `df_plot`.
- **Notebook quirk:** In the red “occluded” load cell, the displayed object is still `df_red_unperturbed` in one line of the source; for publication you may want to verify the occluded frame is what you intend to plot by inspecting the variable passed to `plot()`.

---

## 7. One-sentence link to the next sections of the report

Once \((n_r, n_b, n_g, n_y)\) are fixed as above, the remainder of the report treats **only the symmetric recurrent weights and related maps** as free or analysed quantities, with linearisation and second derivatives evaluated **holding these Hill exponents constant**—except in the optional extension where \(n_g\) itself is promoted to a learnable thirteenth parameter.
