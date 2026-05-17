# Paper 7 — Reviewer Addendum

**Companion to:** `REVIEW.md` (initial review)
**Subject .tex:** `paper2_self_healing_test_automation.tex` (104,932 bytes, last touched 2026-04-28)
**Scope of this addendum:** validate the prior review against the *current* .tex, add findings the prior review missed, and produce a prioritised top-5 fix list.

---

## 1. Validation of prior review against current .tex

| # | Prior major gap | Status | Evidence in current .tex / artifacts |
|---|---|---|---|
| 1 | No experimental baseline comparison | **PARTIALLY ADDRESSED** | §10 (`sec:ablation`, tab:mode_comparison) introduces a `rule_based` WATERFALL-style mode and a `random` mode, but the table still ships with `---` placeholders for those rows (lines 770-773); only `heuristic` and `none` are populated. §13 ("Baseline Comparison") now openly flags this as an external-validity threat. |
| 2 | ML contribution opaque | **FIXED** | §5.1 (`sec:model_diagnostics`, lines 178-204) ships a full model card: AUC 0.9993/0.9994, macro-F1 0.977, precision/recall, TN/FP/FN/TP, training time, 5-fold CV. Feature-importance table (lines 208-231) enumerates all 13 features with weights for both GB and XGB. Three augmentation phases (§5.2-5.3) document feature engineering iterations. |
| 3 | Execution-improvement label/formula mismatch (1.63 vs 2.63×) | **FIXED** | Line 401-405 now defines both forms explicitly: $R = 2.63\times$ (multiplicative factor) and $\Delta_{\text{exec}} = R-1 = 1.63$ (additive gain, 163%). Table 1 still reports the factor form; the disambiguation is clean. |
| 4 | Limited DOM mutation scope (52 mutations, single e-commerce app) | **VALID** | Unchanged: still 52 mutations on the SauceDemo-style storefront. The new "Mutation Realism Audit" (§10.x) maps each class to a citation but does not broaden the workload. |
| 5 | FHR=35.42% under-explained as "deliberate abstention" | **FIXED** | §6 ("False Healing Rate", line 491) and §6.x ("Denominator Decomposition", lines 706-736) now state explicitly that ranking succeeds for 100% of broken events and the entire 35.42% is recovery-executor failure (stale/hidden/detached nodes). The prior misleading "deliberately abstains" framing is gone. |
| 6 | Threshold sweep underdeveloped | **PARTIALLY ADDRESSED** | §6.x (`sec:threshold`, lines 673-704) adds an 8-row conservative sweep with HSR/FHR per threshold; still no per-class threshold and no principled rationale for picking 0.50 over 0.55 — but the sensitivity envelope is now visible. |
| Minor | "logistic regression as the ML fallback" wording | **FIXED** | Line 174 now correctly says "Gradient Boosting (GB) classifier and an XGBoost (XGB) classifier" and justifies tree-ensembles over logistic regression. |
| Minor | Filename `paper2_*.tex` | **VALID** | Not renamed; file is still `paper2_self_healing_test_automation.tex`. |
| Minor | No framework acronym (SHER suggestion) | **VALID** | No acronym introduced anywhere in the .tex. |

**Net assessment:** the largest publication-blocker (ML opaqueness) is fully closed. Baseline comparison is partially closed (infrastructure exists, numbers don't). Mutation breadth is unchanged.

---

## 2. New findings the prior review missed

### 2.1 Reproducibility / DOI formatting

The Zenodo DOI `10.5281/zenodo.19684439` appears in three locations, all formatted consistently as `\url{https://doi.org/10.5281/zenodo.19684439}` (lines 90, 891, 932). **However, the DOI is NOT in the abstract** — the abstract section (lines 31-33) has no DOI, no data-availability sentence, and no mention of the ML/hybrid numbers. ICST/ISSRE reviewers typically expect the artefact DOI either in the abstract or in a first-page footnote. **Recommendation:** add a one-line "Artefacts: doi.org/10.5281/zenodo.19684439" to the abstract or to the IEEEkeywords footer.

### 2.2 Broken cross-reference

Line 880 (conclusion) references `Section~\ref{sec:ml_diagnostics}` but the actual label (line 179) is `sec:model_diagnostics`. This will compile to a "??" in the PDF. **One-character fix.**

### 2.3 ML-improvement number inconsistency (NEW — high severity)

- Line 90 (related work): "machine-learning ranking ablation showing **+8.1\%** HSR improvement over the heuristic-only baseline"
- Line 880 (conclusion): hybrid 68.63% vs heuristic 64.58% → "**+4.05\,pp** improvement"
- `reports/mode_comparison.csv`: heuristic 0.6458, ml 0.6531, hybrid 0.6863

None of these reconcile cleanly. 68.63 − 64.58 = 4.05 pp absolute, or ≈6.3% relative; 65.31 − 64.58 = 0.73 pp; neither produces 8.1%. The 8.1% claim in §2.4 appears to be either stale or computed against an unstated baseline. **Pick one definition, apply globally.**

### 2.4 `multimodel_summary.json` is all zeros

`reports/multimodel_summary.json` lists Heuristic, ML (GB), Hybrid (GB), ML (XGB), Hybrid (XGB) entries with `broken: 0, healed: 0, hsr: 0.0` for *all five*. The paper's claim that "both algorithms converge to the same hybrid HSR (68.63%)" (line 880) is therefore not backed by this artefact. The actual HSR values do live in `mode_comparison.csv` and `ml_ablation_summary.csv`, so the number itself is reproducible, but `multimodel_summary.json` looks like a regeneration script that ran but never wrote real values. **Either fix the script or remove the orphan JSON before Zenodo re-release.**

### 2.5 Figures / tables / refs audit

- All 21 `\includegraphics` paths resolve in `figures/`.
- All 7 `\input{reports/*.tex}` paths resolve in `reports/`.
- All `\ref{tab:*}` and `\ref{fig:*}` resolve **except** `\ref{sec:ml_diagnostics}` on line 880 (see 2.2).
- `tables/` directory exists but is **empty**; all tables now live under `reports/`. Either delete the empty directory or move generated tables into it for tidiness.

### 2.6 Bibliography: 12 of 19 entries are orphans

Only the following 7 bibitems are cited: `arcuri2013` (line 808), `hammoudi2016` (lines 812, 871, 882, plus realism table), `stocco2018` (line 812, realism table), `ricca2019` (realism table), `yandrapally2014` (realism table). The remaining **12 entries are uncited**: `bertolino2021`, `chen2016xgboost`, `choudhary2011`, `friedman2001gb`, `garcia2020`, `grechanik2009`, `javvadi2025defect_dataset`, `javvadi2025healing_dataset`, `leotta2013`, `memon2013`, `mesbah2012`, `panichella2019`, `xie2008`, `selenium`. Notably, the self-citing dataset DOIs (`javvadi2025healing_dataset`, `javvadi2025defect_dataset`) are present as bibitems but the paper instead inlines the bare DOIs via `\url{}` — so the self-citations never actually formally cite themselves. No circular cites detected. No `references.bib` file exists; bibliography is inline.

### 2.7 13-feature vector enumeration

The prior review asked whether the 13 features are enumerated anywhere. **Answer: yes, fully, in `tab:feature_importance`** (lines 208-231), which lists all 13 by name with both GB and XGB importances. This is no longer code-only. The text reference "the thirteen features" (line 252) is supported by the table, though a one-sentence English description grouping them (pairwise vs candidate-intrinsic) would help non-skim readers.

### 2.8 Writing: specific unclear / padded sentences

1. **Line 32 (abstract):** "At runtime, the framework monitors historical execution data and page-structure patterns to generate predictive locator models that enable automated recovery during test execution." — Two redundant "during test execution" clauses; the framework also does not "generate predictive locator models at runtime" (training is offline). Misleading and padded.
2. **Line 174:** "The current implementation uses two interchangeable tree-ensemble rankers as the ML fallback: a Gradient Boosting (GB) classifier and an XGBoost (XGB) classifier." — "Interchangeable" implies symmetry but §5.1 documents a deliberate precision/recall trade-off; either drop the word or replace with "two configurable…".
3. **Line 491:** "the framework is not silently selecting incorrect elements, but it is also not refusing to act; it is committing to a candidate and then failing to interact with it." — Triple-negative, hard to parse. Suggest: "Every unresolved failure occurs *after* a candidate is selected: the executor cannot drive the chosen element."
4. **Line 738 (table caption):** "When a candidate was ranked, how often did it heal?" — coincides numerically with row 1 in the same table, which the text on line 756 then admits ("the first two denominators coincide"). Either collapse the two rows or explain up-front why they are reported separately.
5. **Line 882 (future work):** "Fifth, evaluating on larger and more diverse web applications (React, Angular, Vue with shadow DOM) and on the integrated TestForge AI platform for production deployment." — Run-on; cramps a generalisation study and a production-deployment study into one item.

### 2.9 Cross-paper / acronym

No SHER (or equivalent) acronym introduced. No cross-references to Paper 8 (test-gen) or the integrated TestForge platform anywhere in the body (mentioned only in line 882's future-work bullet). The prior review's cross-paper note still stands.

---

## 3. Prioritised top-5 fix list (effort estimates)

| Rank | Fix | Effort | Why first |
|---|---|---|---|
| 1 | **Populate the empty rows of Table~\ref{tab:mode_comparison}** by running `rule_based` and `random` modes end-to-end (the harness exists, so this is a CLI sweep). Without populated baseline rows, the paper's headline contribution is unanchored. | 4-8 h (compute) + 1 h (table regen) | Highest reviewer-risk gap. Closes the prior review's #1. |
| 2 | **Reconcile the +8.1% vs +4.05 pp inconsistency** (§2.3 above). Pick a single definition of "ML improvement" (absolute pp or relative %), apply globally to abstract, §2.4, §10, and conclusion. Also fix `\ref{sec:ml_diagnostics}` → `sec:model_diagnostics`. | 30 min | Single-number coherence is a publication blocker; reviewers will flag this on first read. |
| 3 | **Add Zenodo DOI to abstract or first-page footnote**; rewrite the abstract's two padded sentences (§2.8 items 1 and 4). Promote the hybrid 68.63% HSR result into the abstract — currently the abstract only reports the weaker 64.58% heuristic number. | 1 h | Abstract is what reviewers anchor on. |
| 4 | **Regenerate `multimodel_summary.json`** (currently all zeros) and either delete the empty `tables/` directory or migrate generated tables into it. Trim the 12 uncited bibitems or actually cite them (chen2016xgboost and friedman2001gb at minimum should be cited where GB/XGB are introduced on line 174). | 2 h | Hygiene; cheap reviewer points. |
| 5 | **Rename file** `paper2_self_healing_test_automation.tex` → `paper7_self_healing_test_automation.tex`, update `run_full_pipeline.py` and any LaTeX include paths. Add SHER (or chosen) acronym in §1. Add 1-paragraph cross-reference to Paper 8 / TestForge platform in §11. | 1 h | Series consistency; carries from prior review. |

Estimated total: **8-12 person-hours** to close the top-5; substantially less than the prior review's "~2 weeks" estimate because the largest item (ML model card) is now done.

---

## 4. Closing note

The .tex has clearly been revised heavily since `REVIEW.md` was written: the ML model card, denominator decomposition, threshold sweep, augmentation phases, candidate-intrinsic features, cost model, mutation realism audit, and bootstrap effect-size section are all new and address most of the prior review's "Major gaps." The two surviving high-severity issues are the unpopulated baseline rows in `tab:mode_comparison` and the +8.1% / +4.05 pp / 0.73 pp inconsistency across abstract, related work, and conclusion. Both are sub-day fixes.
