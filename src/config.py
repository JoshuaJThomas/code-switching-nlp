"""
Project configuration — single source of truth for all constants.

Benchmarking Modern Multilingual Transformer Models for
Malay-English Code-Switched Sentiment Analysis
Joshua Joenathan Thomas (25141571) | Amit Kumar Gupta (25109952)
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Datasets" / "MESocSentiment-main" / "MESocSentiment Corpus"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = DATA_DIR / "Train Set.csv"
TEST_CSV  = DATA_DIR / "Test Set.csv"
FULL_CSV  = DATA_DIR / "MESocSentiment Corpus.csv"

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Label mapping ────────────────────────────────────────────────────────────
LABEL2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

# ─── Data splits ──────────────────────────────────────────────────────────────
# Provided train set is used directly (balanced by original authors via oversampling).
# We deduplicate it, then do a stratified 90/10 train/dev split.
# The 2,000-sample test set (manual labels column) is the benchmark — never touched during training.
DEV_SIZE = 0.10          # 10% of deduplicated train as dev set
TEST_LABEL_COL = "Sentiment (Manual)"   # use human labels from test set

# ─── TF-IDF + SVM ─────────────────────────────────────────────────────────────
SVM_CONFIG = {
    "tfidf_max_features": 50_000,
    "tfidf_ngram_range": (1, 2),   # unigrams + bigrams
    "random_state": SEED,
}

# ─── Transformer fine-tuning (shared hyperparameters) ─────────────────────────
FINETUNE_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,   # reduce to 4 + grad_accum=4 if OOM on 8GB
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 1,    # set to 4 if batch_size reduced to 4
    "learning_rate": 2e-5,
    "warmup_ratio": 0.10,
    "weight_decay": 0.01,
    "max_seq_length": 128,
    "metric_for_best_model": "eval_macro_f1",
    "load_best_model_at_end": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "logging_steps": 50,
    "fp16": False,          # set True on CUDA; MPS uses bf16 or fp32
    "bf16": False,          # set True on Apple MPS or CUDA Ampere+
    "dataloader_num_workers": 0,
}

# ─── Model registry (locked — no additions) ───────────────────────────────────
MODELS = {
    "svm": {
        "name": "TF-IDF + SVM",
        "checkpoint": None,
        "results_dir": RESULTS_DIR / "baseline",
        "note": "Classical ML baseline (sklearn LinearSVC)",
    },
    "xlm_r": {
        "name": "XLM-R",
        "checkpoint": "xlm-roberta-base",
        "results_dir": RESULTS_DIR / "xlm_r",
        "note": "General-purpose multilingual transformer; architectural baseline",
    },
    "mdeberta": {
        "name": "mDeBERTa-v3",
        "checkpoint": "microsoft/mdeberta-v3-base",
        "results_dir": RESULTS_DIR / "mdeberta",
        "note": "Stronger general-architecture baseline. Cite HuggingFace model card.",
    },
    "xlm_t": {
        "name": "XLM-T",
        "checkpoint": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "results_dir": RESULTS_DIR / "xlm_t",
        "note": (
            "Domain-adapted baseline. Already fine-tuned on ~198M tweets for sentiment. "
            "Twitter pretraining + prior sentiment fine-tuning are named confounds — "
            "gains cannot be attributed to architecture alone."
        ),
    },
}

# ─── Phase 3 (LLM denoising — conditional, only after Checkpoint B) ───────────
LLM_CONFIG = {
    "n_ambiguous_samples": 500,
    "prompt_dev_set_size": 30,   # 10 per class, from training data only
    "model": "claude-sonnet-4-6",  # or any capable LLM
    "shots_per_class": 5,
}

# ─── V2 Corrected Experiments ─────────────────────────────────────────────────
# Gold test set (2,000 samples) is split into a clean dev set (for checkpoint
# selection) and a clean test set (for final evaluation). This fixes the critical
# flaw in the original study where checkpoint selection used a noisy dev set,
# causing the transformer to memorise labelling errors rather than true semantics.
CLEAN_DEV_SIZE = 300                                       # absolute count from gold 2,000
CLEAN_DEV_CSV  = RESULTS_DIR / "test_clean_dev_split.csv"   # 300 gold-clean dev samples
CLEAN_TEST_CSV = RESULTS_DIR / "test_clean_test_split.csv"  # 1,700 gold-clean test samples

# ─── Cross-Corpus Datasets (Phase 4 — dataset compliance) ────────────────────
DUCK_DIR          = ROOT / "Datasets" / "dUCk" / "Annotated The dUCk Tweets Dataset"
DUCK_FULL_TRAIN_XML = DUCK_DIR / "Full Training Dataset"       / "full_training_dataset.xml"
DUCK_FULL_TEST_XML  = DUCK_DIR / "Full Testing Dataset"        / "full_testing_dataset.xml"
DUCK_CS_TRAIN_XML   = DUCK_DIR / "Code-Switching Training Dataset" / "eng_malay_training_dataset.xml"
DUCK_CS_TEST_XML    = DUCK_DIR / "Code-Switching Testing Dataset"  / "eng_malay_testing_dataset.xml"

CROSS_CORPUS_DIR  = RESULTS_DIR / "cross_corpus"

# ─── SQLite database path (shared across all notebooks) ───────────────────────
DB_PATH = RESULTS_DIR / "mesocsentiment.db"

FINETUNE_CONFIG_V2 = {
    **FINETUNE_CONFIG,              # inherit all base settings
    "num_train_epochs": 7,          # was 3 — more epochs + clean dev enables proper convergence
    "max_grad_norm": 1.0,           # gradient clipping; critical for mDeBERTa stability
}

MODELS_V2 = {
    "xlm_r": {
        **MODELS["xlm_r"],
        "results_dir": RESULTS_DIR / "xlm_r_v2",
    },
    "mdeberta": {
        **MODELS["mdeberta"],
        "results_dir": RESULTS_DIR / "mdeberta_v2",
    },
    "xlm_t": {
        **MODELS["xlm_t"],
        "results_dir": RESULTS_DIR / "xlm_t_v2",
    },
    "xlm_r_focal_g1": {
        "name": "XLM-R (Focal γ=1.0)",
        "checkpoint": "xlm-roberta-base",
        "results_dir": RESULTS_DIR / "xlm_r_focal_g1",
        "note": "XLM-R with Focal Loss (γ=1.0) — ablation for class-imbalance correction.",
    },
    "xlm_r_focal_g2": {
        "name": "XLM-R (Focal γ=2.0)",
        "checkpoint": "xlm-roberta-base",
        "results_dir": RESULTS_DIR / "xlm_r_focal_g2",
        "note": "XLM-R with Focal Loss (γ=2.0) — ablation for class-imbalance correction.",
    },
}
