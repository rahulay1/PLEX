# PLEX × ModernBERT on GoEmotions

Efficient, perturbation-free local explanations for text classifiers.
This repo pairs **ModernBERT** (`cirimus/modernbert-base-go-emotions`) with a lightweight **PLEX** head (Siamese projection + cosine) to mimic token-level importance from LIME—**without** costly perturbations at test time.

* 🧠 Backbone: ModernBERT (base, 28 GoEmotions labels)
* 🪄 Explainer head: PLEX (projection + cosine similarity over (CLS, token) embeddings)
* 🔬 Supervision: LIME token importances (train-time only)
* ⚡ Inference: Fast, single forward pass (no perturbations)
* 🧪 Notebook: end-to-end data → features → targets → training → eval → demo

---

## Contents

* `modernbert_base_go_emotions.ipynb` — end-to-end notebook (data processing, training, evaluation, demo)
* `plex_seismic_modernbert_lime.pth` — trained PLEX head (produced by the notebook)
* (Optional) `goemotions_*_with_lime_words.pt` — cached features + LIME targets

---

## Quick Start

> Python ≥ 3.9 recommended. GPU strongly recommended for LIME; PLEX inference is fast on CPU.

```bash
# 1) Create a clean env (example with venv)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps (adjust CUDA wheel if needed; CPU-only works too)
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 lime==0.2.0.1 matplotlib==3.8.4 pandas==2.2.2 scipy==1.13.1 tqdm==4.66.4

# 3) Launch Jupyter and open the notebook
jupyter notebook modernbert_base_go_emotions.ipynb
```

---

## What the Notebook Does

1. **Load data** (GoEmotions `test.tsv`: `text`, `labels`, `meta`)
2. **Make splits** (default: **400** train / **200** test sentences)
3. **Extract embeddings**

   * Forward ModernBERT with `output_hidden_states=True`
   * Merge subwords → words via `offset_mapping` (average subword vectors)
   * Save CLS + word embeddings
4. **Generate LIME targets**

   * Use ModernBERT’s **top predicted class** per sentence
   * Run LIME for that class and **align** to whitespace words
5. **Train PLEX head**

   * Pairs (CLS, word) → target importance
   * Weighted SmoothL1 + (optional) small pairwise rank loss
   * Token saliency thresholding to reduce label noise
6. **Evaluate**

   * Sentence-level **Spearman** & **Top-k overlap** (PLEX vs LIME)
   * **Stress test**: Δprob drop after removing top-k tokens (LIME vs PLEX) with 95% CI
7. **Demo**

   * Enter your sentence → plot **PLEX** vs **LIME** word heatmaps

---

## Key Config (edit in notebook)

```python
MODEL_NAME  = "cirimus/modernbert-base-go-emotions"
LAYER_IDX   = -1        # try 9 for mid-layer features
TRAIN_N     = 400
TEST_N      = 200
EPOCHS      = 10
LR          = 7e-4
BATCH_SIZE  = 2048
TAU         = 0.05      # drop low-magnitude LIME targets
RANK_WEIGHT = 0.2       # 0.3 to emphasize ordering
MARGIN      = 0.2
SEED        = 42
```

---

## Using the Trained Head on New Sentences

After running the notebook you’ll have `plex_seismic_modernbert_lime.pth`.
Use the provided **demo** section in the notebook or a small script to:

* predict top label with ModernBERT
* compute PLEX importances (single pass)
* run LIME (optional, for comparison)
* plot side-by-side heatmaps

---

## Interpreting Metrics

* **Spearman (median):** overall rank agreement (PLEX vs LIME) across tokens
* **Top-k overlap:** fraction of shared top-k important words
* **Stress test Δprob:** drop in ModernBERT’s predicted probability after removing top-k words chosen by each explainer (**higher = more influential**). Comparable means (overlapping 95% CIs) ⇒ **faithful** token selection by PLEX at a fraction of LIME’s cost.

---

## Troubleshooting

* **HF 401 Unauthorized:** `huggingface-cli login` or clear a stale `HF_TOKEN`.
* **PyTorch 2.6 `torch.load(weights_only=...)`:** load Python objects with `weights_only=False`.
* **CUDA OOM:** lower `BATCH_SIZE`, try `LAYER_IDX=9`, or run CPU (slower but safe).
* **LIME slowness:** reduce `num_features` or parallelize per sentence. Remember LIME is **train-time only**.
* **Token alignment quirks:** ensure extraction uses `offset_mapping` and avoids truncation (two-pass encode with `max_length=seq_len`).

---

## Reproducibility

* Fixed `SEED=42` for NumPy & PyTorch.
* Minor nondeterminism may occur across hardware; metrics remain close.

---

## Extending

* **Layers:** set `LAYER_IDX=9` (often sharper token saliency than last layer).
* **Backbones:** swap `MODEL_NAME` (same embedding dim) and re-run steps 3–5.
* **Targets:** swap LIME with SHAP or **ensemble** (e.g., `λ·SHAP + (1−λ)·LIME`).
* **Uncertainty:** enable MC-dropout at inference for variance estimates.

---

## Citation

If you use this repo, please cite:

* **GoEmotions:**
  Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
  *GoEmotions: A Dataset of Fine-Grained Emotions.*
  [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions)

* **ModernBERT fine-tune:**
  Junqué de Fortuny, E. (2025). *Emotion Detection with ModernBERT.*
  [https://huggingface.co/cirimus/modernbert-base-go-emotions](https://huggingface.co/cirimus/modernbert-base-go-emotions)

* **PLEX (this work):**
  *[Add your paper citation / arXiv link here]*

---

## License

*Add your license (e.g., MIT) here. If using GoEmotions/ModernBERT, respect their licenses and terms.*

---

## Acknowledgements

Thanks to the maintainers of **Transformers**, **LIME**, and the creators of **GoEmotions** and the **ModernBERT** fine-tune used here.

---
