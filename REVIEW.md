# Paper 7 — Reviewer Report

**Title:** AI-Driven Self-Healing Test Automation: A DOM-Similarity and Machine-Learning Framework for Resilient Web UI Regression Testing
**Filename note:** the .tex is named `paper2_self_healing_test_automation.tex` for historical reasons; this is **Paper 7** of the 9-paper series.
**Target venue (inferred):** ICST / ISSRE / ICSE main (IEEEtran conference, ~13 pages)

**Verdict: MAJOR (~2 weeks of fixes)** — solid empirical pipeline, exemplary reproducibility, but two publication-blocking issues: no experimental baseline comparison and ML contribution opaque (despite "machine learning" in title).

---

## Headline claims

* Aggregate HSR = 64.58% with Wilson 95% CI [62.65%, 66.47%] over n = 2,400 broken-locator events.
* Hybrid mode HSR = 68.63%, +4.05pp over heuristic baseline.
* 13-feature DOM similarity vector.
* Hybrid threshold = 0.50, α = 0.40.
* Per-class asymmetry: id_change recovers at 86.2%, attribute_removed at 31.6% (55-pt spread).
* +8.1% HSR improvement reported in abstract (ML over heuristic).

## Strengths

1. **Reproducibility artefact excellence** — full mutation pipeline, all analysis scripts, per-run step logs, regenerable CSVs/figures. Zenodo DOI 10.5281/zenodo.19684439 cited correctly. README + REPRODUCIBILITY.md comprehensive.
2. **Statistical rigor** — Wilson CIs (not naive), bootstrap SE = 0.0100, per-run σ = 0.043, χ² + Kruskal-Wallis + Friedman tests.
3. **Mutation taxonomy grounded in practice** — seven classes match industrial refactoring patterns; imbalanced distribution (18 id_change vs 1 placeholder) is realistic.
4. **Transparent ablation** — heuristic vs rule_based vs random vs ml vs hybrid; mode_comparison.csv shows progression: 64.58% → 65.31% (ml) → 68.63% (hybrid).
5. **Honest per-class asymmetry reporting** — id_change 86.2% vs attribute_removed 31.6%; the 55-pt spread explicitly explains the aggregate.

## Major gaps

1. **No experimental baseline comparison** — paper claims "no peer-reviewed evaluations exist for ROBULA+, WATERFALL, Healenium" and treats this as a strength. ICSE/ICST reviewers will ask: did you reimplement WATERFALL's rule-based fallback (ID → name → placeholder → aria_label → class+tag) and run it on the same dataset? If not, the 64.58% number is unanchored.
2. **ML contribution opaque** — title promises "Machine Learning"; the paper text says "logistic regression as the ML fallback" (line 174). The actual code uses **Gradient Boosting** (and an XGBoost variant). README claims AUC 0.9993 (GB) / 0.9994 (XGB) — these numbers appear nowhere in the paper. No model card, no train/test protocol, no AUC, no precision/recall, no confusion matrix.
3. **Execution-improvement inconsistency** — line 298 formula: ΔExec = (950+1550)/950 − 1 ≈ **1.63** (63% improvement). Table 1 line 331: "Execution improvement factor ≈ **2.63×** baseline." 1.63 = ratio − 1; 2.63 = full ratio. Wording / labelling inconsistent.
4. **Limited DOM mutation scope** — 52 mutations × 5 versions × single e-commerce app. Class change (1), placeholder (1), reorder (4) undersampled. No real-world refactoring scenarios (Vue → React, Bootstrap → Tailwind).
5. **False-healing rate of 35.42% under-explained** — paper says framework "deliberately abstains" but recovery-executor decomposition (§5.2) shows 100% of broken events produced a ranked candidate. So FHR is **failed recovery execution**, not abstention. Wording misleads.
6. **Threshold sweep underdeveloped** — Table 3 shows HSR vs threshold but no principled selection rationale for 0.50. Per-mutation threshold tuning not attempted.

## Minor issues

* Citation: Leotta et al. cited as 2016 in abstract; actual ICSE-SEIP 2016 confirmed.
* "Locator recovery accuracy" numerically equals HSR (line 285) — paper acknowledges but the dual term confuses. Rename one.
* `threshold_sweep.png` would benefit from confidence-band shading.
* Line 174 says "logistic regression" — code says GB. Fix.
* Cross-paper: framework has no acronym (unlike RAITG in paper 8). Consider naming (e.g., SHER = Self-Healing Element Recovery).

## Cross-paper coupling

* Shared Zenodo DOI cited correctly.
* Identified as "paper 7 of a 9-paper TestForge AI series."
* No cross-references to paper 8 (test-gen) — they share the TestForge platform; should reference each other in deployment / integration discussion.
* Filename `paper2_*.tex` is a historical artefact — rename to `paper7_self_healing_test_automation.tex` to match the series ordering and avoid reviewer confusion.

## Concrete fix list

1. **`healing/ml_ranker.py:174` and paper §X.Y:** replace "logistic regression as the ML fallback" with "Gradient Boosting classifier (with optional XGBoost variant)." Cite hyperparameters (max_depth, n_estimators, learning_rate).
2. **New §4.4 ML Model Card:** add ROC AUC = 0.9993 (GB), 0.9994 (XGB), F1, precision@0.50, confusion matrix, feature importance top-3, train/test protocol (5-fold stratified CV on 2,400 events).
3. **§3 Related Work (new "Baseline Reimplementation" paragraph):** describe WATERFALL-style rule-based fallback (ID → name → placeholder → aria_label → class+tag); add it as a fifth mode in `mode_comparison.csv`. Report HSR. (If short on time: add it as explicit threat in §T-T-V instead of skipping.)
4. **§5.2:** decompose 850 unresolved failures: "X ranked but executor failed (executor bottleneck); Y ranked below threshold (ranking bottleneck)."
5. **Table 1 line 331:** change "Execution improvement factor ≈ 2.63× baseline" to "Total executions / baseline = 2.63 (i.e., 63% additional executions enabled)." OR change formula text to match 2.63. Pick one and apply throughout.
6. **Appendix A: ML model card** — CV strategy, AUC, feature importance, train/test protocol; figure (importance bar chart).
7. **Abstract line 90:** soften — "to our knowledge, none have published independent peer-reviewed empirical evaluations" (current language risks reviewer pushback).
8. **§5 Discussion line 384:** rewrite "deliberately abstains" — clarify the executor vs ranker bottleneck distinction (per fix 4).
9. **New §6.X Threshold sweep:** report ablation on threshold ∈ {0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70} × {ml, hybrid}; per-class threshold sensitivity.
10. **Filename:** rename `paper2_self_healing_test_automation.tex` → `paper7_self_healing_test_automation.tex`; update build scripts.
11. **Cross-paper:** add 1-paragraph reference to paper 8's test-case generation as orthogonal contribution; mention the integrated TestForge platform.
12. **Zenodo DOI:** verified canonical; no change needed.
