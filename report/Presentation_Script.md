# Programming for AI — Project Presentation Script
**Team:** Joshua Joenathan Thomas (25141571) | Amit Kumar Gupta (25109952)  
**Duration:** 10 minutes | **Format:** Video recording

---

## SLIDE STRUCTURE & SPEAKER NOTES

---

### SLIDE 1 — Title Slide (0:00–0:30) | **Josh**

**Slide content:**
- Title: *Benchmarking Multilingual Transformer Models for Malay–English Code-Switched Sentiment Analysis*
- Authors and student IDs
- Module: Programming for AI | NCI | April 2026

**Script:**
> "Hello, my name is Joshua Thomas and I'm presenting alongside my teammate Amit Gupta. Our project is a controlled benchmark study of sentiment analysis methods on a linguistically unusual type of data: Malay–English code-switched Twitter posts. Over the next ten minutes we'll explain why this problem is worth solving, how we approached it, what we found, and what it means."

---

### SLIDE 2 — The Problem (0:30–1:30) | **Josh**

**Slide content:**
- Example tweet: *"Memang best la that concert, totally worth it lah!"*
- Map: Malaysian Twitter usage statistics
- Why it matters: brand monitoring, public health, political analysis
- The challenge: tools built for English don't understand Malay

**Script:**
> "Malaysian Twitter users routinely mix Malay and English within a single sentence — a phenomenon called code-switching. A post like 'Memang best la that concert' blends Malay sentiment words with English, and no English-only tool can reliably interpret it. This matters commercially: brand monitoring, public health surveillance, and political analysis in Malaysia all require tools that actually work on this kind of text. Our project asks: which class of model performs best here, and by how much?"

---

### SLIDE 3 — Dataset & Key Challenge (1:30–2:30) | **Amit**

**Slide content:**
- MESocSentiment corpus: 19,714 tweets, 2,000 manually annotated gold test (primary benchmark)
- dUCk Group Tweets: 2,256 Malay–English brand tweets, XML format (cross-corpus validation)
- tweet_eval (SemEval-2017): 57,899 English Twitter posts (monolingual reference corpus)
- SentiLexM (26,004 entries) + MELex (6,132 entries) — bilingual lexicons
- Class distribution chart: 69.3% Neutral, 19.4% Negative, 11.3% Positive (MESocSentiment)
- The 33.8% label noise finding — semi-automatic training labels vs gold
- SQLite database: all 5 datasets stored before any model evaluation (14 tables)

**Script:**
> "We used five datasets in total, all stored in a shared SQLite database before any processing. The primary corpus is MESocSentiment: 9,906 training samples and 2,000 manually annotated gold test tweets. When we compared those training labels to the gold annotations, we found a 33.8% disagreement rate — a third of training labels potentially wrong, acting as a ceiling on all supervised methods. We also incorporated the dUCk Group Tweets dataset — 2,256 Malay-English brand tweets in XML format — for cross-corpus validation, and tweet_eval, 57,899 English-only Twitter posts, as a monolingual reference. This three-corpus design lets us show whether our findings generalise beyond a single dataset."

---

### SLIDE 4 — Our Approach: Three Tiers (2:30–3:30) | **Amit**

**Slide content:**
- Tier diagram: Zero-Shot → Classical Supervised → Transformer Fine-Tuned
- Models listed per tier (13 total)
- Key methodological corrections highlighted (clean dev/test split, 7 epochs, Focal Loss)

**Script:**
> "We evaluated 13 model configurations across three tiers. Zero-shot baselines — VADER, TextBlob, and pysentimiento, plus two bilingual lexicons we constructed. A classical supervised TF-IDF plus LinearSVC. And seven fine-tuned transformer configurations including XLM-R, XLM-T, mDeBERTa, and Focal Loss variants. A critical correction from standard practice: we used a 300-sample clean development set drawn from the manually annotated data for checkpoint selection — not the noisy training pool. This single change significantly affects how well the checkpoints generalise."

---

### SLIDE 5 — Main Results Table (3:30–5:00) | **Josh**

**Slide content:**
- Table with all 13 results (the same as Table 1 in the paper)
- Colour-coded by tier
- Arrows showing the two key gaps

**Script:**
> "Here are our main results. The most important finding is visible immediately: the largest gap is not where people expect it. Going from the best zero-shot tool — pysentimiento at 0.48 — to the SVM at 0.64, is a jump of over 17 percentage points. That's more than ten times the size of the gap from the SVM to our best transformer model, which adds only 1.8 percentage points. The dominant barrier in this domain is not architectural sophistication. It's the presence or absence of labelled training data. A simple TF-IDF model trained on 9,000 tweets massively outperforms a 278-million parameter model being used out of the box."

---

### SLIDE 6 — Focal Loss & Statistical Confidence (5:00–6:30) | **Amit**

**Slide content:**
- Focal Loss γ=1.0 result: Macro-F1 = 0.6841 (single run), mean = 0.6743
- CI chart: XLM-R std [0.6473, 0.6801] vs Focal γ=1.0 [0.6427, 0.7059]
- Why Focal Loss helps: Positive class only 11.3% of test set
- Per-class F1 comparison: standard XLM-R vs Focal γ=1.0

**Script:**
> "Our best result comes from XLM-R with Focal Loss, a loss function that dynamically down-weights easy majority-class examples and forces the model to focus on hard minority-class ones. The Positive class is only 11 percent of the test set, so standard cross-entropy tends to ignore it. Focal Loss brought the Positive-class F1 from 0.60 to 0.63. To verify stability, we ran three seeds and computed confidence intervals using the t-distribution — appropriate for n equals 3. The CI lower bound of 0.6427 lies just above the SVM baseline of 0.6407, confirming the transformer advantage is real but narrow."

---

### SLIDE 7 — The Denoising Experiment (6:30–7:45) | **Josh**

**Slide content:**
- Phase 3 design: Shannon entropy → 1,500 high-entropy samples → LLM relabelling → retrain
- Result: Macro-F1 = 0.6305, Δ = −0.0282 (null/negative result)
- Explanation diagram: LLM annotator Macro-F1 = 0.600 on gold test

**Script:**
> "We ran a supplementary experiment asking: can we improve performance by using an LLM to clean the noisy training labels? We selected the 1,500 training samples the fine-tuned model was most uncertain about, and asked Claude Sonnet to relabel them using a culturally grounded few-shot prompt. The result was a decrease of 2.8 percentage points. Why? When we tested Claude Sonnet directly on the gold test set, it achieved only Macro-F1 of 0.60 — lower than the fine-tuned model's 0.66. A model that performs worse than our baseline is not a reliable oracle for correcting the labels our baseline already finds difficult. The denoising redistributed noise rather than removing it."

---

### SLIDE 8 — Methodological Pitfalls (7:45–8:45) | **Amit**

**Slide content:**
- Four pitfalls listed with before/after Macro-F1 where measurable
- mDeBERTa-v3 failure timeline (3 configs, training loss ≥ 4.37)
- The v1 → v2 delta table

**Script:**
> "A major contribution of this work is a methodological audit. We identified four systematic design flaws that affect code-switching NLP benchmarks generally — not just ours. Noisy development sets, insufficient training epochs, misidentified training failures, and impotent class-imbalance treatment. mDeBERTa-v3 exhibited consistent training collapse across three configurations, with training loss stuck above 4.37 from the first step — when the expected loss for a random classifier is 1.1. The failure is reproducible on our hardware setup, though the root cause — a likely interaction between the ContextPooler architecture and CUDA precision — was not fully verified. We document it honestly as a negative result with a plausible but unconfirmed hypothesis."

---

### SLIDE 9 — Conclusions & Impact (8:45–9:40) | **Josh**

**Slide content:**
- Key takeaway: zero-shot → supervised gap >> supervised → transformer gap
- Cross-corpus finding: English tools recover on English Twitter (tweet_eval) — confirming the challenge is code-switching, not the models
- Deployment implication: TF-IDF + SVM is a serious contender for resource-constrained deployment
- Future work: full corpus re-annotation, 5+ seed variance, Malay-domain pre-trained models
- Societal relevance: Malaysia's growing multilingual social media landscape

**Script:**
> "The headline conclusion is a practical one: for Malay-English code-switched sentiment analysis, a well-tuned TF-IDF plus SVM captures most of the available performance at a fraction of the inference cost of a transformer. The additional 1.8 points from XLM-R may not justify 278 million parameters in production. Our cross-corpus evaluation confirmed this specifically: VADER's Macro-F1 jumps from 0.38 on MESocSentiment to 0.53 on English-only Twitter — a 38% relative improvement — and pysentimiento rises from 0.48 to 0.72, a 51% gain. These numbers prove that English lexicon tools are not fundamentally broken; they are missing Malay vocabulary. The SVM shows the opposite pattern, dropping from 0.64 in-domain to 0.46 on English Twitter, confirming that its TF-IDF vocabulary is corpus-specific. Future work should prioritise full corpus re-annotation — the 33.8% noise rate is the primary performance ceiling — and evaluation of models pre-trained on Malay-domain data."

---

### SLIDE 10 — Thank You / Questions (9:40–10:00) | **Both**

**Slide content:**
- Summary: 13 models, 3 tiers, 4 pitfalls, 1 null result, 1 clean benchmark
- Student IDs
- "Questions?"

**Script (Josh):**
> "To summarise: we benchmarked 13 configurations across three tiers on a clean test set, identified and corrected four systematic methodological flaws, and delivered a result that challenges the assumption that transformers are automatically the best choice for low-resource code-switched NLP. Thank you."

---

## SLIDE DECK NOTES

**Suggested slide count:** 10 slides  
**Recommended tool:** PowerPoint or Google Slides  
**Key visuals to prepare:**
1. Example code-switched tweet with annotation
2. Class distribution bar chart (from `results/baseline/corpus_distribution.png`)
3. Three-tier architecture diagram (pyramid or staircase)
4. Results table (colour-coded: zero-shot = red, SVM = amber, transformer = green)
5. CI interval chart comparing XLM-R std vs Focal γ=1.0 vs SVM baseline
6. Phase 3 denoising pipeline flowchart
7. Four methodological pitfalls summary table
8. Cross-corpus Macro-F1 bar chart (`results/cross_corpus/cross_corpus_comparison.png` — generated by notebook 01d)

**Timing breakdown:**
| Section | Speaker | Time |
|---|---|---|
| Title + Problem | Josh | 0:00–1:30 |
| Dataset + Approach | Amit | 1:30–3:30 |
| Main Results | Josh | 3:30–5:00 |
| Focal + CI | Amit | 5:00–6:30 |
| Denoising | Josh | 6:30–7:45 |
| Pitfalls | Amit | 7:45–8:45 |
| Conclusions | Josh | 8:45–9:40 |
| Sign-off | Both | 9:40–10:00 |

**Total: 10:00 exactly**
