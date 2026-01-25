## STAR (Trajectory STAR) — Pointwise Drift Alignment for HC-SOINN

This document describes the **STAR method used in this repository**: **Trajectory STAR** (pointwise drift alignment with EMA), tightly integrated with **HC-SOINN**.

---

## Motivation

In class-incremental learning, the backbone (feature extractor) is updated after each task:

- Old task space: \(f_{t-1}(\cdot)\)
- New task space: \(f_t(\cdot)\)

This causes **feature drift** for old classes. Instead of constraining the backbone (e.g., distillation), STAR follows the philosophy:

- **Adapt rather than resist**: keep updating the backbone for new tasks
- **Move the classifier/prototypes** to match the new feature space

Trajectory STAR avoids fitting a global rigid transform (rotation/scale/translation). Instead, it uses **the observed drift trajectory of anchor samples** and applies it **directly to the corresponding HC-SOINN nodes**.

---

## Objects and Notation

- **Backbone / feature extractor** at task \(t\): \(f_t(x)\in\mathbb{R}^D\)
- **HC-SOINN nodes** (per class \(c\)):
  - raw (unnormalized) center: \(p^{raw}_{c,i}\in\mathbb{R}^D\) (`center_raw`)
  - normalized center (for cosine inference): \(p_{c,i}=\text{normalize}(p^{raw}_{c,i})\) (`center`)
- **NCM class center** (per class \(c\)):
  - raw: \(\mu^{raw}_c\) (`class_mu_raw[c]`)
  - normalized: \(\mu_c=\text{normalize}(\mu^{raw}_c)\) (`class_mu[c]`)
- **Anchor image** paired with node \(i\): \(x_{c,i}\)
- **Anchor drift** between tasks:
  - instant drift: \(\Delta_{c,i}=f_t(x_{c,i})-f_{t-1}(x_{c,i})\)
  - EMA drift: \(\widehat{\Delta}_{c,i}\)
- EMA coefficient: \(\lambda\in(0,1]\) (config: `star_lambda`)

---

## Workflow (per task)

Trajectory STAR has two phases around each task transition.

### A) Anchor selection (after finishing Task \(t\), for current/new classes)

For each newly introduced class \(c\):

1. Obtain all HC-SOINN nodes \(\{p^{raw}_{c,i}\}_{i=1}^{M_c}\).
2. For each node \(i\), select **one anchor image** \(x_{c,i}\) from the class dataset:
   - maximize cosine similarity between \(p^{raw}_{c,i}\) and the sample feature
   - resolve collisions (two nodes picking the same sample) by selecting the next-best candidate
3. Store:
   - anchor images \(\{x_{c,i}\}\)
   - reference features \(\{f_t(x_{c,i})\}\)
   - EMA buffer \(\widehat{\Delta}_{c,i}\leftarrow 0\)
   - a snapshot of node centers for robust matching in later tasks

**Key requirement**: anchors are stored **1:1 with nodes**.

### B) Alignment (before evaluating Task \(t+1\), for old classes)

For each old class \(c\):

1. Re-extract features for stored anchor images using the current backbone:
   \[
   f_{t+1}(x_{c,i})
   \]
2. Compute instant drift per node:
   \[
   \Delta_{c,i}=f_{t+1}(x_{c,i})-f_t(x_{c,i})
   \]
3. EMA smooth the drift:
   \[
   \widehat{\Delta}_{c,i}\leftarrow (1-\lambda)\widehat{\Delta}_{c,i}+\lambda\Delta_{c,i}
   \]
4. Update each node’s raw center by **pointwise transport**:
   \[
   p^{raw}_{c,i}\leftarrow p^{raw}_{c,i}+\widehat{\Delta}_{c,i}
   \]
   and refresh normalized center:
   \[
   p_{c,i}\leftarrow \text{normalize}(p^{raw}_{c,i})
   \]
5. Update NCM center using the **mean node drift**:
   \[
   \mu^{raw}_c \leftarrow \mu^{raw}_c + \frac{1}{M_c}\sum_{i=1}^{M_c}\widehat{\Delta}_{c,i}
   \quad,\quad
   \mu_c\leftarrow \text{normalize}(\mu^{raw}_c)
   \]
6. Chain update:
   - set stored reference features to the current ones:
     \[
     f_t(x_{c,i}) \leftarrow f_{t+1}(x_{c,i})
     \]

---

## Inference (unchanged)

Inference continues to use cosine-based classification over normalized representations:

- nodes use `center` (normalized)
- NCM uses `class_mu` (normalized)

Trajectory STAR only updates `center_raw` / `class_mu_raw` and then re-normalizes.

---

## Algorithm (pseudocode)

```text
Given:
  Feature extractor f_t
  HC-SOINN nodes p_{c,i}(raw) for each class c
  NCM centers mu_c(raw)
  EMA coefficient λ in (0, 1]

After task t (for new classes):
  for each new class c:
    for each node i in class c:
      choose anchor image x_{c,i} closest to node center (cosine), avoid duplicates
      store feats_ref_{c,i} = f_t(x_{c,i})
      store ema_delta_{c,i} = 0

Before evaluation at task t+1 (for old classes):
  for each old class c:
    for each stored anchor i:
      feat_new = f_{t+1}(x_{c,i})
      delta = feat_new - feats_ref_{c,i}
      ema_delta_{c,i} = (1-λ)*ema_delta_{c,i} + λ*delta
      p_{c,i}(raw) = p_{c,i}(raw) + ema_delta_{c,i}
      p_{c,i} = normalize(p_{c,i}(raw))
      feats_ref_{c,i} = feat_new

    mu_c(raw) = mu_c(raw) + mean_i(ema_delta_{c,i})
    mu_c = normalize(mu_c(raw))
```

---

## Hyperparameters

- **`star_mode`**: set to `"trajectory"` to enable Trajectory STAR.
- **`star_lambda`** (\(\lambda\)): EMA coefficient controlling drift update speed.
  - smaller (e.g. 0.2): smoother / more conservative updates
  - larger (e.g. 0.7–1.0): faster adaptation but more sensitive to noise

---

## Implementation mapping (this repo)

- Core implementation: `utils/STAR.py`
  - anchor selection: `STARAligner.select_anchors_for_current_task(...)`
  - alignment: `STARAligner.align_old_classes(...)`
  - `anchor_store[cls]` holds:
    - `images`: anchor images (1:1 with nodes)
    - `feats_ref`: stored reference features
    - `ema_delta`: per-node EMA drift
    - `centers_raw_ref`: reference raw centers for node matching across tasks
- Learner wiring:
  - `models/dualprompt.py`, `models/coda_prompt.py`, `models/aper_adapter.py`
  - these pass `star_mode` and `star_lambda` into `STARAligner(...)`

---

## Determinism note (important for fair comparison)

HC-SOINN’s internal SOINN refinement used to shuffle signals with python’s global RNG, which could make Task0 results differ across runs when other components consumed random numbers.

This repo makes that refinement shuffle deterministic by using a local RNG in `utils/hc_soinn_classifier.py`, so enabling/disabling STAR does not silently change Task0 HC-SOINN prototypes.








