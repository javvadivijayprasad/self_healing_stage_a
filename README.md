# Self-Healing Test Automation — Stage A

Artifact repository for the paper **"AI-Driven Self-Healing Test Automation Using
Machine Learning for Adaptive UI Test Recovery."** The framework recovers broken
Selenium locators by capturing the live DOM, extracting candidate elements,
scoring them with a weighted heuristic over tag, text, attribute, class, parent,
and DOM-depth similarity, and retrying the failed step with the top-ranked
candidate when its score crosses a configurable threshold.

Stage A contains the complete heuristic engine, a mutation generator that
synthesises broken UI versions from a reference baseline, the Selenium test
harness, and every analysis script that produces the tables and figures in the
paper.

---

## Repository layout

```
self_healing_stage_a/
├── app_versions/                 # Five HTML UI versions (v1 baseline + v2–v5 mutated)
├── dom_breaker/                  # DOM mutation generator
│   └── dom_mutation_generator.py
├── healing/                      # Self-healing engine
│   ├── self_heal_engine.py       #   find_element with baseline → heal fallback
│   ├── heuristic_ranker.py       #   weighted similarity ranker (threshold = 0.45)
│   ├── rule_based_ranker.py      #   attribute-fallback ablation baseline
│   ├── random_ranker.py          #   random same-tag ablation baseline
│   ├── ranker_factory.py         #   create_ranker(mode, ...)
│   ├── candidate_extractor.py
│   ├── dom_capture.py
│   ├── locator_store.py
│   └── utils.py
├── tests/                        # Eight Selenium test cases
├── experiment/                   # Analysis, figures, LaTeX table generators
├── reports/                      # Generated CSVs, text reports, .tex tables
├── figures/                      # Generated PNGs
├── run_experiment.py             # Entry point for a single experiment run
├── run_full_pipeline.py          # Full 16-step pipeline (mutations → stats)
├── requirements.txt
├── REPRODUCIBILITY.md            # Step-by-step reproduction guide
├── LICENSE                       # MIT
└── README.md
```

---

## Quick start

Requires Python 3.10+, Chrome, and an internet connection (the first run
downloads a matching ChromeDriver via `webdriver-manager`).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# Serve the five UI versions on http://localhost:8000
python -m http.server 8000 --directory app_versions

# In a second terminal: run the full pipeline
python run_full_pipeline.py
```

On completion the pipeline produces:

- `reports/results.csv` — per-run, per-test outcome table
- `reports/locator_healing_report.csv` — per-step healing events
- `reports/mutation_healing_summary.csv` — per-mutation-class recovery
- `reports/statistical_analysis.{csv,txt,tex}` — Wilson CIs, χ², Kruskal–Wallis, Friedman
- `reports/threshold_sweep.{csv,tex}` — conservative HSR vs. acceptance threshold
- `reports/denominator_decomposition.csv` — ranked-and-resolved vs. ranked-but-unresolved vs. no-candidate
- `reports/mode_comparison.{csv,tex}` — ranker ablation (if multiple modes have been run)
- `figures/*.png` — every figure in the paper

---

## Ranker ablation

Every ranker is selected by a single CLI flag. Non-default modes also write a
tagged copy of the results so comparisons can run without overwriting the
canonical file.

```bash
python run_experiment.py --healer-mode heuristic  --runs 50   # produces reports/results.csv
python run_experiment.py --healer-mode rule_based --runs 50   # produces reports/results_rule_based.csv
python run_experiment.py --healer-mode random     --runs 50   # produces reports/results_random.csv
python run_experiment.py --healer-mode none       --runs 50   # healing disabled
python experiment/generate_mode_comparison.py                 # emits ablation table + figure
```

| Mode          | Description                                                            |
|---------------|------------------------------------------------------------------------|
| `heuristic`   | Default. Weighted similarity ranker with threshold 0.45.               |
| `rule_based`  | Attribute fallback (ID → name → placeholder → aria_label → class+tag → tag). |
| `random`      | Random same-tag candidate. Lower-bound sanity baseline.                |
| `ml`/`hybrid` | Reserved for the trained ML ranker; currently falls back to heuristic. |
| `none`        | Healing disabled. Reports the baseline failure rate under mutation.    |

---

## Key entry points

| Script                                                | Purpose                                                                 |
|-------------------------------------------------------|-------------------------------------------------------------------------|
| `dom_breaker/dom_mutation_generator.py`               | Generate versions 2–5 from the baseline HTML.                           |
| `healing/analyze_mutations.py`                        | Classify each mutation by type and locator.                             |
| `run_experiment.py`                                   | Run the Selenium suite under one healer mode.                           |
| `experiment/generate_statistical_analysis.py`         | Wilson CI, χ², Kruskal–Wallis, Friedman, bootstrap SE.                  |
| `experiment/generate_threshold_sweep.py`              | Conservative HSR/FHR vs. acceptance threshold + denominator split.      |
| `experiment/generate_mutation_vs_healing_table.py`    | Per-mutation-class recovery table and figure.                           |
| `experiment/generate_mode_comparison.py`              | Ranker ablation comparison.                                             |

See **REPRODUCIBILITY.md** for a step-by-step guide that reproduces every
number in the paper from a clean checkout.

---

## Citation

If you use this artifact, please cite the accompanying paper:

```bibtex
@inproceedings{janakiraman2026selfheal,
  title     = {AI-Driven Self-Healing Test Automation Using Machine Learning
               for Adaptive UI Test Recovery},
  author    = {Janakiraman, Vijay Prasad},
  booktitle = {Proceedings of the IEEE},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
