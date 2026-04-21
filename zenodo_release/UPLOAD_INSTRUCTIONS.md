# Zenodo Upload Instructions
## Creating a New Record for Self-Healing Test Automation Dataset

This is a **separate Zenodo record** from the defect prediction dataset (10.5281/zenodo.19682733). The two datasets are cross-referenced but have independent DOIs.

---

## Step 1 -- Create New Upload

1. Go to **https://zenodo.org/uploads/new**
2. Log in with your account
3. Select **Upload type: Dataset**

---

## Step 2 -- Upload Files

Upload the following files (or package as a zip):

### Experiment Results:
```
results_heuristic.csv              (~4 MB)
results_ml_gb.csv                  (~4.7 MB)
results_ml_xgb.csv                 (~4.5 MB)
results_hybrid_gb.csv              (~4.9 MB)
results_hybrid_xgb.csv             (~4.5 MB)
mutation_report.csv
```

### Training Data:
```
healing_training_data.csv          (1.1 MB, 7,750 labeled examples)
```

### Event Logs:
```
heal_events.csv                    (3.0 MB, 13,000+ candidate records)
heal_decisions.csv                 (2.6 MB, 5,500+ decision summaries)
```

### Trained Models:
```
healing_ranker_gb.pkl              (612 KB)
healing_ranker_gb.metrics.json
healing_ranker_xgb.pkl             (217 KB)
healing_ranker_xgb.metrics.json
model_comparison.json
```

### Paper:
```
self_healing_paper.pdf             (2.4 MB, 31 pages)
self_healing_paper.tex
```

### Metadata:
```
README.md                          (copy of DATA_README.md)
CITATION.cff
```

> **Tip:** You can zip everything into `self-healing-test-automation-dataset-v1.0.0.zip` for a single upload.

---

## Step 3 -- Fill in Metadata

Copy these values from `.zenodo.json`:

| Field | Value |
|-------|-------|
| **Title** | Self-Healing Test Automation Dataset: 11,100 Experiment Rows with DOM-Similarity ML Models |
| **Authors** | Javvadi, Vijay P. / Independent Researcher |
| **Description** | *(paste from .zenodo.json description field)* |
| **Version** | 1.0.0 |
| **License** | Creative Commons Attribution 4.0 International |
| **Keywords** | self-healing test automation, UI testing, Selenium, DOM similarity, machine learning, gradient boosting, XGBoost, hybrid ranking, locator recovery, mutation testing, software quality, empirical software engineering |
| **Publication date** | 2026-04-21 |

### Related identifiers:
1. `https://github.com/javvadivijayprasad/self-healing-test-automation` → **is supplement to**
2. `10.5281/zenodo.19682733` → **is part of** (links to defect prediction dataset)

---

## Step 4 -- Publish

1. Click **Save** (saves as draft)
2. Review the preview
3. Click **Publish**

> Zenodo assigns a new DOI. Note it down and update CITATION.cff and DATA_README.md with the real DOI.

---

## Step 5 -- Post-Publish Updates

### Update self_healing_paper.tex with Data Availability:

```latex
\section*{Data Availability}
The experimental dataset, trained models (Gradient Boosting and XGBoost),
and reproducibility scripts described in this paper are publicly available
on Zenodo at \url{https://doi.org/10.5281/zenodo.XXXXXXX} (version 1.0.0)
under a Creative Commons Attribution 4.0 International licence.
```

### Cross-reference from defect prediction record:

Consider adding a note to the defect prediction record (10.5281/zenodo.19682733) pointing to this new dataset as a companion.

### Update GitHub README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

---

## Checklist Before Publishing

- [ ] New upload created at zenodo.org/uploads/new
- [ ] All CSV files uploaded (5 result files + mutation_report + training data + events)
- [ ] Both .pkl models uploaded (GB + XGBoost)
- [ ] model_comparison.json uploaded
- [ ] self_healing_paper.pdf uploaded
- [ ] README.md uploaded (from DATA_README.md)
- [ ] CITATION.cff uploaded
- [ ] Title, description, keywords filled from .zenodo.json
- [ ] Version set to 1.0.0
- [ ] License set to CC BY 4.0
- [ ] GitHub repo added as related identifier (is supplement to)
- [ ] Defect prediction DOI added as related identifier (is part of)
- [ ] Preview reviewed before clicking Publish
- [ ] DOI noted and updated in CITATION.cff and DATA_README.md after publishing

---

## Why a Separate Record (Not a New Version)

| Approach | Pros | Cons |
|----------|------|------|
| **Separate record (chosen)** | Own DOI, clean citation, correct data scope | Two DOIs to track |
| Same record (version 2.0.0) | Single DOI | Mixes unrelated datasets, confusing for citers |

The datasets have completely different schemas, domains, and research questions. Separate records with cross-references is the Zenodo best practice for related but distinct datasets. The `isPartOf` relation links them in Zenodo's metadata graph.

---

## EB-1A Relevance

Two separate Zenodo datasets with cross-references strengthens the EB-1A case:
- **Two distinct research contributions** (defect prediction + self-healing) with independent citations
- **Both link to the same research series**, showing breadth and depth
- **Total data:** 284,676 defect prediction instances + 11,100 self-healing experiment rows
- **Independent DOIs** mean each dataset can accumulate its own citation count
