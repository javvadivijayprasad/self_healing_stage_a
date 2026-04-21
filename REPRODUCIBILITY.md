# Reproducibility Guide

This guide reproduces every table, figure, and statistic reported in the paper
**"AI-Driven Self-Healing Test Automation Using Machine Learning for Adaptive
UI Test Recovery"** from a clean checkout of this repository.

All commands assume the repository root (`self_healing_stage_a/`) as the
working directory.

---

## 1. Environment

| Component       | Version used for the paper                                |
|-----------------|-----------------------------------------------------------|
| OS              | Windows 11 (22H2); also verified on Ubuntu 22.04          |
| Python          | 3.11.x                                                    |
| Chrome          | 120+ (any version supported by the current ChromeDriver)  |
| Selenium        | `requirements.txt` pins                                   |
| webdriver-manager | fetches a matching ChromeDriver on first run            |
| NumPy / Pandas / Matplotlib | `requirements.txt` pins                         |

> `scipy` is **not** required. The statistical test implementations in
> `experiment/generate_statistical_analysis.py` use numpy only: Wilson score
> CIs, the Wilson–Hilferty approximation of the χ² survival function, an
> inline average-rank Kruskal–Wallis H, and a Friedman test.

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows PowerShell / cmd
# source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

---

## 2. Serve the five UI versions

The experiment drives a local web server that hosts the baseline (`version_1`)
and four mutated versions (`version_2`…`version_5`) under
`app_versions/`:

```bash
python -m http.server 8000 --directory app_versions
```

Leave this running in one terminal.

---

## 3. Reproduce the main results (heuristic ranker)

In a second terminal:

```bash
python run_full_pipeline.py
```

This runs the full 16-step pipeline:

| Step | Script                                                  | Emits                                             |
|------|---------------------------------------------------------|---------------------------------------------------|
| 1    | `dom_breaker/dom_mutation_generator.py`                 | `app_versions/version_{2..5}/`                    |
| 2    | `healing/analyze_mutations.py`                          | `reports/mutation_report.csv`                     |
| 3    | `run_experiment.py`                                     | `reports/results.csv`                             |
| 4    | (pipeline step: dataset validation)                     | —                                                 |
| 5    | `healing/analyze_results.py`                            | summary in `reports/`                             |
| 6    | `experiment/generate_paper_figures.py`                  | `figures/*.png`                                   |
| 7    | `experiment/generate_architecture_figures.py`           | architecture PNGs                                 |
| 8    | `experiment/generate_locator_healing_report.py`         | `reports/locator_healing_report.csv`              |
| 9    | `experiment/generate_locator_fragility.py`              | fragility table / figure                          |
| 10   | `experiment/generate_locator_similarity_heatmap.py`     | `figures/locator_similarity_heatmap.png`          |
| 11   | `experiment/generate_mutation_vs_healing_table.py`      | `reports/mutation_healing_summary.csv`, `.tex`    |
| 12   | `experiment/generate_advanced_analysis.py`              | top-failed-tests, healing efficiency              |
| 13   | `experiment/generate_experiment_summary.py`             | `reports/experiment_summary.{csv,txt}`            |
| 14   | `experiment/generate_statistical_analysis.py`           | Wilson CI, χ², Kruskal–Wallis, Friedman tables    |
| 15   | `experiment/generate_threshold_sweep.py`                | threshold sweep, denominator decomposition        |
| 16   | `experiment/generate_mode_comparison.py`                | ablation (no-op unless multiple modes have run)   |

Expected wall time on a developer laptop: **~45–60 minutes** (50 experiment
runs × 4 mutated versions × 8 tests).

---

## 4. Reproduce the ranker ablation

The ablation table in Section VI of the paper compares four ranker modes.
Every mode writes a mode-tagged results file so runs do not overwrite one
another.

```bash
python run_experiment.py --healer-mode heuristic  --runs 50
python run_experiment.py --healer-mode rule_based --runs 50
python run_experiment.py --healer-mode random     --runs 50
python run_experiment.py --healer-mode none       --runs 50
python experiment/generate_mode_comparison.py
```

Output:

- `reports/results.csv` (heuristic, canonical)
- `reports/results_rule_based.csv`
- `reports/results_random.csv`
- `reports/results_none.csv`
- `reports/mode_comparison.csv`
- `reports/mode_comparison_table.tex`
- `figures/mode_comparison.png`

---

## 5. Reproduce the threshold sweep

Already produced by Step 15 of `run_full_pipeline.py`, but can be re-run
independently after `results.csv` and `locator_healing_report.csv` exist:

```bash
python experiment/generate_threshold_sweep.py
```

Emits `reports/threshold_sweep.{csv,tex}`,
`reports/denominator_decomposition.csv`, and
`figures/{threshold_sweep,denominator_decomposition}.png`.

The sweep is a **conservative reclassification**: at each threshold `T`, every
heal whose `healing_score < T` is reclassified as a failure. This yields a
lower-bound HSR curve without re-running the experiment.

---

## 6. Determinism and variance

- `run_experiment.py` re-seeds Python's `random` at the top of every run with
  `time.time()`, so individual test-ordering is non-deterministic. To obtain
  a deterministic sweep, set `PYTHONHASHSEED=0` and patch the `random.seed`
  call to a fixed integer.
- The aggregate HSR is stable across runs within the Wilson 95% CI width
  reported in `reports/statistical_analysis.txt`. The bootstrap SE (1000
  resamples) is reported on the same line.

---

## 7. Expected artifacts for a paper rebuild

The LaTeX source in `paper2_self_healing_test_automation.tex` imports:

```
reports/mutation_healing_summary.tex
reports/statistical_analysis_table.tex
reports/threshold_sweep_table.tex
reports/mode_comparison_table.tex
figures/healing_success_rate.png
figures/threshold_sweep.png
figures/denominator_decomposition.png
figures/hsr_per_version_ci.png
figures/hsr_per_test_ci.png
figures/mutation_failure_impact.png
figures/mode_comparison.png
```

If any of these are missing, re-run the corresponding generator from the
tables in Sections 3–4 above.

---

## 8. Troubleshooting

| Symptom                                              | Fix                                                                    |
|------------------------------------------------------|------------------------------------------------------------------------|
| `SessionNotCreatedException: ... Chrome version`     | Let `webdriver-manager` refresh: delete `~/.wdm/` and rerun.           |
| `Address already in use :8000`                       | Another http.server is running; stop it or use `--bind 127.0.0.1 8001` and update `BASELINE_ROOT_URL` in `run_experiment.py`. |
| `results.csv missing column: ...`                    | A partial run did not finish STEP 3. Delete `reports/results.csv` and re-run. |
| Pipeline STEP 16 prints *"no results files found"*   | Expected when only one mode has been run. Run the ablation sweep (Section 4). |

---

## 9. License

Code and analysis scripts are released under the [MIT License](LICENSE). The
HTML UI versions under `app_versions/` are derived from a public SauceDemo-
style clone and are distributed for reproducibility only.
