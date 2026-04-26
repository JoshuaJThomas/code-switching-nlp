QUICK START
-----------
  1. cd into this repository root
  2. pip install -r requirements.txt
  3. jupyter notebook  (or jupyter lab)
  4. Open notebooks/ and run in numbered order

  Public results are pre-populated. Raw datasets and the generated SQLite
  database are not redistributed in this public repository.

  Requires internet for:  tweet_eval download (notebook 01d, HuggingFace automatic)
                          transformer checkpoints (~1-3 GB each, notebooks 02+)
  Requires GPU for:       notebooks 02, 02_focal_loss, 02_focal_multiseed, 04
  Requires API key for:   notebooks 04, 05
                          (set env var: export ANTHROPIC_API_KEY=<your key>)

================================================================================
  SUBMISSION PACKAGE — Malay-English Sentiment Analysis Benchmarking
  Joshua Joenathan Thomas (25141571) | Amit Kumar Gupta (25109952)
  Module: Programming for AI | NCI | April 2026
================================================================================

PUBLIC REPOSITORY CONTENTS
--------------------------
report/
  Programming_in_AI.pdf         - Final report
  Report_Thomas_Gupta.docx      - Editable report copy, if present
  Presentation_Script.md        - Slide-by-slide speaker notes for 10-minute video

notebooks/                      - Run in the numbered order below
  01_data_pipeline_and_svm.ipynb
  01b_lexicon_baselines.ipynb
  01c_bilingual_lexicon.ipynb
  01d_cross_corpus_eval.ipynb
  02_transformer_v2.ipynb
  02_focal_loss.ipynb
  02_focal_multiseed.ipynb
  03_results_analysis.ipynb
  04_llm_denoising_v2.ipynb
  05_llm_benchmark.ipynb

src/
  config.py          - Single source of truth for all paths and hyperparameters

requirements.txt     - Python dependencies

results/
  all_results.json              - Ground-truth results for all 13 model configs
  analysis_summary.json         - CI values, std, and key statistics
  results_table.csv             - Human-readable results table
  macro_f1_comparison.png       - Overall Macro-F1 bar chart
  per_class_f1_comparison.png   - Per-class F1 heatmap
  training_curves.png           - Transformer training/validation curves
  cross_corpus/
    cross_corpus_results.json   - Cross-corpus eval numbers (6 models x 3 corpora)
    cross_corpus_comparison.png - Bar chart comparing corpora
    duck_eda.png                - dUCk dataset EDA
    teval_eda.png               - tweet_eval EDA

EXECUTION ORDER (MANDATORY)
----------------------------
Notebooks must be run in this exact sequence. Each notebook populates the
SQLite database locally and/or saves artefacts that later notebooks depend on.

  Step 1: 01_data_pipeline_and_svm.ipynb
          Creates SQLite DB, loads MESocSentiment, runs TF-IDF+SVM.
          Output: mesocsentiment.db, results/baseline/

  Step 2: 01b_lexicon_baselines.ipynb
          VADER, TextBlob, pysentimiento zero-shot evaluation.
          Output: results/lexicons/

  Step 3: 01c_bilingual_lexicon.ipynb
          SentiLexM + MELex bilingual lexicon construction and evaluation.
          Requires internet (downloads lexicons from GitHub if not cached).
          Stores sentilexm and melex tables to SQLite.
          Output: results/bilingual_lexicon/

  Step 4: 01d_cross_corpus_eval.ipynb
          Loads dUCk Group Tweets (XML) and tweet_eval (HuggingFace).
          Evaluates all zero-shot + SVM models across 3 corpora.
          Stores duck_* and tweet_eval_* tables to SQLite.
          Requires internet (HuggingFace datasets).
          Output: results/cross_corpus/

  Step 5: 02_transformer_v2.ipynb
          Fine-tunes XLM-R, mDeBERTa-v3, XLM-T with v2 methodology
          (clean dev split, 7 epochs, gradient clipping).
          REQUIRES: GPU (tested on RTX 4060 Ti 16 GB). ~4 hours runtime.
          Output: results/xlm_r_v2/, results/mdeberta_v2/, results/xlm_t_v2/

  Step 6: 02_focal_loss.ipynb
          XLM-R with Focal Loss (gamma=1.0 and gamma=2.0).
          REQUIRES: GPU. ~2.5 hours runtime.
          Output: results/xlm_r_focal_g1/, results/xlm_r_focal_g2/

  Step 7: 02_focal_multiseed.ipynb
          3-seed stability run for XLM-R v2 and Focal gamma=1.0.
          Computes t-distribution 95% CI (t=4.303, df=2, sample std).
          Output: results/focal_multiseed/

  Step 8: 03_results_analysis.ipynb
          Aggregates all results, generates comparison charts.
          Run AFTER all model notebooks are complete.
          Output: results/all_results.json, results/results_table.csv, charts

  Step 9: 04_llm_denoising_v2.ipynb
          Phase 3: entropy-based sample selection, Claude API relabelling,
          retrain and evaluate. REQUIRES: ANTHROPIC_API_KEY env variable.
          Phase 3 is a null/negative result (Macro-F1 dropped 0.0282).
          Output: results/phase3_v2/

  Step 10: 05_llm_benchmark.ipynb
          Evaluates Claude as a direct annotator on the gold test set.
          REQUIRES: ANTHROPIC_API_KEY env variable.
          Output: results/llm_benchmark/


KEY RESULTS SUMMARY
--------------------
  Best robust model: XLM-R (Focal gamma=1.0)  Mean Macro-F1 = 0.6748
                                                95% CI [0.6664, 0.6832]
                                                Best single run = 0.6841
  TF-IDF + SVM:                               Macro-F1 = 0.6407
  Best zero-shot:    pysentimiento             Macro-F1 = 0.4778
  Gap (zero->SVM):                            +0.1629 (34.1% relative gain)
  Gap (SVM->best):                            +0.0434 (narrower than expected)

  Cross-corpus (code-switching effect):
    VADER:           MESocSentiment 0.3827 -> tweet_eval 0.5262 (+38%)
    pysentimiento:   MESocSentiment 0.4778 -> tweet_eval 0.7188 (+51%)
    TF-IDF+SVM:      MESocSentiment 0.6407 -> tweet_eval 0.4592 (-28%)

  Phase 3 (LLM denoising): Macro-F1 = 0.6305, Delta = -0.0282 (null result)


ENVIRONMENT SETUP
------------------
  Python 3.10+
  pip install -r requirements.txt

  Environment variables needed for Phase 3 / LLM notebooks:
    ANTHROPIC_API_KEY = <your key>

  The notebooks use relative imports from src/config.py.
  Make sure the working directory is the project root when running:
    cd <project_root>
    jupyter notebook


ARTEFACT AUDIT
---------------
  All numerical results in the report were verified against all_results.json
  and analysis_summary.json before publication.

================================================================================
