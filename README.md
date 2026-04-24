# Code-Switching NLP

Joint research project (Thomas & Gupta) exploring code-switching language detection and sentiment analysis using multiple ML approaches — from classical baselines through to transformer fine-tuning and LLM benchmarking.

## Overview

Code-switching — alternating between languages mid-sentence — is a major challenge for NLP systems. This project benchmarks a progression of approaches:

1. **Data pipeline + SVM** — baseline classical classifier
2. **Bilingual lexicon** — lexicon-based baseline
3. **Focal loss transformer** — fine-tuned transformer with focal loss to handle class imbalance
4. **Multi-seed evaluation** — robustness testing across seeds
5. **Cross-corpus evaluation** — generalisation across datasets
6. **LLM denoising** — using LLMs to clean noisy code-switched text
7. **LLM benchmarking** — comparing LLM zero-shot performance

## Tech Stack
- Python 3.x
- Transformers (HuggingFace)
- Scikit-learn
- PyTorch (focal loss)
- Pandas / NumPy
- Matplotlib

## Structure
```
notebooks/     # All experiment notebooks (run in order — see README_RUN_ORDER.txt)
src/           # Config and shared utilities
results/       # F1 charts, training curves, cross-corpus results, JSON metrics
report/        # Joint research paper (docx) + presentation script
requirements.txt
README_RUN_ORDER.txt
```

## Results
See `results/` for macro F1 comparisons, per-class F1, training curves, and cross-corpus evaluation plots.

## Run
```bash
pip install -r requirements.txt
# See README_RUN_ORDER.txt for notebook execution order
jupyter notebook notebooks/
```

---
MSc Artificial Intelligence — NCI Dublin, 2025
Joint work: Joshua Thomas & Amit Gupta
