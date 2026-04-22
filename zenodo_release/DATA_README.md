# Self-Healing Web Test Automation Dataset
## 11,100 Experiment Rows with DOM-Similarity and ML-Driven Locator Recovery

**Version:** 1.0.0
**DOI:** *(assigned after Zenodo upload)*
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Authors:** Vijay P. Javvadi (research@vijayjavvadiresearch.ai)
**Date released:** April 2026

---

## Overview

This dataset supports the research paper "AI-Driven Self-Healing Test Automation: A DOM-Similarity and Machine-Learning Framework for Resilient Web UI Regression Testing." It contains the complete experimental data, trained models, mutation configurations, and reproducibility scripts for a self-healing web test automation framework that automatically recovers broken UI locators using DOM similarity analysis and machine-learning ranking.

**Total experiment rows:** 11,100 (across 5 ranking configurations)
**Broken locator events:** 2,400+ per mode
**Healed events:** Up to 1,750 per mode
**Candidate feature records:** 13,000+ (from continuous-learning event logger)
**ML training instances:** 7,750 labeled examples
**Mutation events:** 52 across 5 UI versions
**Test suites:** 9 Selenium-based suites
**Ranking configurations:** 5 (heuristic, ML-GB, ML-XGB, hybrid-GB, hybrid-XGB)

**Companion dataset:** The defect prediction dataset for Papers 1-6 of this research series is available separately at DOI [10.5281/zenodo.19682733](https://doi.org/10.5281/zenodo.19682733).

---

## Experimental Results Summary

| Mode | Algorithm | Broken | Healed | Failed | HSR | 95% CI |
|------|-----------|--------|--------|--------|-----|--------|
| Heuristic | --- | 2,400 | 1,550 | 850 | 64.6% | [62.6, 66.5] |
| ML | Gradient Boosting | 2,450 | 1,600 | 850 | 65.3% | [63.4, 67.2] |
| ML | XGBoost | 2,350 | 1,450 | 900 | 61.7% | [59.7, 63.6] |
| Hybrid | Gradient Boosting | 2,550 | 1,750 | 800 | 68.6% | [66.8, 70.4] |
| Hybrid | XGBoost | 2,550 | 1,750 | 800 | 68.6% | [66.8, 70.4] |

---

## File Contents

```
experiment_results/
  results_heuristic.csv              -- Heuristic baseline (1,600 rows)
  results_ml_gb.csv                  -- ML with Gradient Boosting (1,600 rows)
  results_ml_xgb.csv                 -- ML with XGBoost (1,600 rows)
  results_hybrid_gb.csv              -- Hybrid with Gradient Boosting (1,600 rows)
  results_hybrid_xgb.csv             -- Hybrid with XGBoost (1,600 rows)
  mutation_report.csv                -- All 52 mutation events with metadata

training_data/
  healing_training_data.csv          -- 7,750 labeled training examples (13 features + label)

heal_events/
  heal_events.csv                    -- 13,000+ per-candidate feature records (continuous learning log)
  heal_decisions.csv                 -- 5,500+ per-decision summaries

models/
  healing_ranker_gb.pkl              -- Trained Gradient Boosting classifier
  healing_ranker_gb.metrics.json     -- GB model performance summary
  healing_ranker_xgb.pkl             -- Trained XGBoost classifier
  healing_ranker_xgb.metrics.json    -- XGBoost model performance summary
  model_comparison.json              -- Head-to-head GB vs XGBoost comparison

figures/
  ml_ablation_comparison.png         -- 3-mode HSR comparison (heuristic, ML, hybrid)
  ml_ablation_per_version.png        -- Per-version HSR breakdown
  ml_score_distribution.png          -- Score distributions across ranking modes
  model_comparison.png               -- GB vs XGBoost classification metrics
  multimodel_ablation.png            -- 5-configuration HSR comparison
  multimodel_per_version.png         -- Per-version multi-model breakdown
  healing_success_rate.png           -- Overall HSR bar chart
  healing_vs_failure.png             -- Healed vs failed breakdown
  version_comparison.png             -- Per-version healing performance
  execution_improvement.png          -- Baseline vs healed interactions
  cumulative_healing.png             -- Cumulative healing over executions
  locator_fragility.png              -- Per-locator fragility ranking
  locator_similarity_heatmap.png     -- Pairwise locator similarity
  per_run_hsr_bootstrap.png          -- Bootstrap 95% CI distribution
  mutations_per_page.png             -- Mutation density across pages

tables/
  ml_ablation_table.tex              -- LaTeX: 3-mode ablation
  multimodel_ablation_table.tex      -- LaTeX: 5-configuration ablation
  model_comparison_table.tex         -- LaTeX: GB vs XGBoost metrics
  healing_efficiency_table.tex       -- LaTeX: per-version/per-test HSR
  statistical_analysis_table.tex     -- LaTeX: chi-square and Kruskal-Wallis
  bootstrap_effect_size_table.tex    -- LaTeX: permutation test + Cliff's delta
  cost_model_table.tex               -- LaTeX: maintenance hours avoided
  threshold_sweep_table.tex          -- LaTeX: threshold calibration
  mode_comparison_table.tex          -- LaTeX: heuristic mode comparison
  mutation_density_table.tex         -- LaTeX: mutation density analysis
  mutation_realism_table.tex         -- LaTeX: mutation realism audit

app_versions/
  version_1/                         -- Baseline (unmutated) web application
  version_2/ through version_5/      -- Progressively mutated versions

tests/
  test_inventory.py                  -- 9 Selenium test suites
  test_cart.py
  test_checkout.py
  test_login.py
  test_navigation.py
  test_home_navigation.py
  test_products_listing.py
  test_multi_add_to_cart.py

scripts/
  run_full_pipeline.py               -- 26-step reproducibility pipeline
  run_experiment.py                   -- Single-mode experiment runner
  train_multimodel.py                -- Dual-algorithm training script
  hyperparameter_sweep.py            -- Grid search over both algorithms
  accumulate_training_data.py        -- Continuous learning data accumulator
  extract_training_data.py           -- Feature extraction from experiment logs

paper/
  self_healing_paper.tex             -- Full LaTeX source
  self_healing_paper.pdf             -- Compiled 31-page paper

zenodo_release/
  DATA_README.md                     -- This file
  CITATION.cff                       -- Standard citation file
  .zenodo.json                       -- Zenodo metadata
  UPLOAD_INSTRUCTIONS.md             -- Step-by-step upload guide
```

---

## Feature Schema (13-Feature ML Vector)

Each candidate element is scored using 13 similarity features comparing the original locator metadata to the candidate DOM element:

| Feature | Type | Description |
|---------|------|-------------|
| `tag_match` | binary | 1 if HTML tag matches original (e.g., both `<input>`) |
| `text_similarity` | float [0,1] | SequenceMatcher ratio of visible text content |
| `attribute_similarity` | float [0,1] | SequenceMatcher ratio of concatenated attributes |
| `class_similarity` | float [0,1] | SequenceMatcher ratio of CSS class names |
| `parent_similarity` | float [0,1] | 1 if parent tag matches, 0 otherwise |
| `depth_similarity` | float [0,1] | 1 - |depth_original - depth_candidate| / max_depth |
| `id_similarity` | float [0,1] | SequenceMatcher ratio of element IDs |
| `name_similarity` | float [0,1] | SequenceMatcher ratio of name attributes |
| `placeholder_similarity` | float [0,1] | SequenceMatcher ratio of placeholder text |
| `aria_label_similarity` | float [0,1] | SequenceMatcher ratio of aria-label attributes |
| `type_attr_match` | binary | 1 if type attribute matches (e.g., both `type="text"`) |
| `sibling_count_ratio` | float [0,1] | min(siblings_orig, siblings_cand) / max(...) |
| `multi_attr_match_count` | float [0,1] | Fraction of non-empty attributes that match exactly |

**Label:** `label = 1` for the candidate selected and successfully healed; `label = 0` for all other candidates.

---

## Mutation Categories

| Mutation Type | Count | Description |
|---------------|-------|-------------|
| id_change | 18 | Element ID renamed with version suffix (e.g., `add-backpack` -> `add-backpack-v`) |
| attribute_removed | 14 | Key attribute completely removed from element |
| dom_wrap | 9 | Element wrapped in additional container div |
| text_change | 5 | Visible text content modified |
| element_reorder | 4 | Element moved to different position among siblings |
| class_change | 1 | CSS class name changed |
| placeholder_change | 1 | Placeholder attribute text modified |
| **Total** | **52** | Across 5 progressively mutated versions |

---

## Trained Models

### Gradient Boosting (Primary)
```
Algorithm:  GradientBoostingClassifier + SMOTE (scikit-learn ImbPipeline)
Parameters: n_estimators=200, learning_rate=0.1, max_depth=4
Features:   13 DOM similarity features (see schema above)
AUC:        0.9993
F1-macro:   0.9768
Precision:  0.9614
Recall:     0.9645
CV AUC:     0.9995 +/- 0.0000 (5-fold stratified)
Training:   6,200 instances (80/20 split)
```

### XGBoost
```
Algorithm:  XGBClassifier + SMOTE
AUC:        0.9994
F1-macro:   0.9772
Precision:  1.0000 (zero false positives)
Recall:     0.9290
CV AUC:     0.9995 +/- 0.0000 (5-fold stratified)
Training:   6,200 instances (80/20 split)
Training time: 0.14s (3.5x faster than GB)
```

Loading the models:

```python
import pickle, pandas as pd

with open('models/healing_ranker_gb.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open('models/healing_ranker_xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

features = ['tag_match', 'text_similarity', 'attribute_similarity',
            'class_similarity', 'parent_similarity', 'depth_similarity',
            'id_similarity', 'name_similarity', 'placeholder_similarity',
            'aria_label_similarity', 'type_attr_match',
            'sibling_count_ratio', 'multi_attr_match_count']

df = pd.read_csv('training_data/healing_training_data.csv')
gb_proba = gb_model.predict_proba(df[features])[:, 1]
xgb_proba = xgb_model.predict_proba(df[features])[:, 1]
```

---

## Reproducing the Experiments

### Full pipeline (26 steps, all artefacts)
```bash
python run_full_pipeline.py --with-multimodel
```

### Individual experiments
```bash
# Heuristic baseline
python run_experiment.py --healer-mode heuristic

# ML with Gradient Boosting
python run_experiment.py --healer-mode ml

# Hybrid with Gradient Boosting
python run_experiment.py --healer-mode hybrid

# ML with XGBoost (requires --results-file for separate output)
python run_experiment.py --healer-mode ml --results-file reports/results_ml_xgb.csv

# Hybrid with XGBoost
python run_experiment.py --healer-mode hybrid --results-file reports/results_hybrid_xgb.csv
```

### Train both models
```bash
python scripts/train_multimodel.py --use-best
```

### Requirements
```
Python 3.10+
selenium, webdriver-manager
scikit-learn, imbalanced-learn, xgboost
numpy, matplotlib
Chrome/Chromium browser
```

---

## Related Datasets

| Dataset | Papers | DOI |
|---------|--------|-----|
| Software Defect Prediction (284,676 instances) | Papers 1-6 | [10.5281/zenodo.19682733](https://doi.org/10.5281/zenodo.19682733) |
| Self-Healing Test Automation (11,100 rows) | Paper 7 | *(this dataset)* |

Both datasets are part of a seven-paper AI-driven quality engineering research series by Vijay P. Javvadi.

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{javvadi2026selfhealing_dataset,
  author    = {Javvadi, Vijay P.},
  title     = {Self-Healing Web Test Automation Dataset: 11,100 Experiment
               Rows with DOM-Similarity and ML-Driven Locator Recovery},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.19684439},
  url       = {https://doi.org/10.5281/zenodo.19684439}
}
```

And the companion paper:

```bibtex
@article{javvadi2026selfhealing,
  author  = {Javvadi, Vijay P.},
  title   = {AI-Driven Self-Healing Test Automation: A DOM-Similarity and
             Machine-Learning Framework for Resilient Web UI Regression Testing},
  year    = {2026}
}
```

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share and adapt this material for any purpose, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.

The web application under test is a synthetic e-commerce application created for this study. No production data, user information, or proprietary code is included.

---

## Contact

Vijay P. Javvadi -- research@vijayjavvadiresearch.ai
LinkedIn: linkedin.com/in/vijayjavvadi
Website: vijayjavvadiresearch.ai
