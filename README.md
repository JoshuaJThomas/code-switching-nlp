# Code-Switching NLP: Malay-English Sentiment Analysis

Sentiment analysis on Malay-English code-switched social media text. The model has to handle mid-sentence language switches, noisy tweet-style writing, and roughly 34% label noise in the data — which makes even the baselines harder than they look.

We ran through everything: lexicon tools, SVM, fine-tuned transformers with focal loss, then tested whether an LLM could skip the fine-tuning entirely. It couldn't.

## What we found

| Approach | Macro F1 |
|---|---|
| Lexicon best (pysentimiento) | 0.478 |
| SVM | 0.641 |
| XLM-R baseline | 0.659 |
| XLM-R + focal loss (γ=1.0) | **0.684** |
| Claude Sonnet few-shot | 0.600 |

The jump from lexicons to SVM (+0.16) was bigger than the jump from SVM to a fine-tuned transformer (+0.02). Focal loss helped most with the POSITIVE class, which was consistently the hardest to classify across every approach.

LLM denoising turned out to be a mild negative — pre-processing with Claude to clean noisy text before training dropped F1 by 0.028. The transformer had apparently learned to use the code-switching noise as signal. Cross-corpus evaluation on dUCk and TweetEval showed the models generalise reasonably across datasets.

Multi-seed runs (seeds 42, 123, 456) confirmed the focal loss results hold up: mean macro F1 0.675 ± 0.003.

## Datasets

- **MESocSentiment** — Malay-English code-switched social media posts, 3-class sentiment
- **dUCk** — annotated code-switching tweet subset (~444 test instances)

Both are in `Datasets/`. The `results/mesocsentiment.db` is a SQLite cache of intermediate model outputs used in the analysis notebook.

## Notebooks

Run in order — see `README_RUN_ORDER.txt` for the exact sequence.

```
01_data_pipeline_and_svm.ipynb      — data pipeline + SVM baseline
01b_lexicon_baselines.ipynb         — VADER, TextBlob, pysentimiento, SentiLexM, MELex
01c_bilingual_lexicon.ipynb         — bilingual lexicon approach
01d_cross_corpus_eval.ipynb         — evaluation across MESocSentiment, dUCk, TweetEval
02_focal_loss.ipynb                 — XLM-R with focal loss
02_focal_multiseed.ipynb            — multi-seed robustness check
02_transformer_v2.ipynb             — XLM-R, mDeBERTa, XLM-T comparisons
03_results_analysis.ipynb           — all plots and tables
04_llm_denoising_v2.ipynb           — LLM-based text denoising pre-processing
05_llm_benchmark.ipynb              — Claude Sonnet few-shot evaluation
```

## Stack

Python 3, HuggingFace Transformers, PyTorch, scikit-learn, pandas, matplotlib

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Report

`report/Programming_in_AI.pdf` — full write-up with methodology, results tables, and analysis.

---

MSc Artificial Intelligence, NCI Dublin 2025  
Joshua Thomas & Amit Gupta
