"""Bootstrap confidence intervals, permutation tests, and Cliff's delta.

All computed from the existing reports/results.csv. No new experiments
are required. These are non-parametric robustness checks requested by
reviewers familiar with empirical software engineering methodology
(Arcuri & Briand 2014).

Outputs:
  - reports/bootstrap_effect_size.csv
  - reports/bootstrap_effect_size.txt
  - reports/bootstrap_effect_size_table.tex
  - figures/per_run_hsr_bootstrap.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"

RESULTS_FILE = REPORT_DIR / "results.csv"


def per_run_hsr(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run_id", "ui_version"], as_index=False).agg(
        healed=("healed_steps", "sum"),
        failed=("failed_steps", "sum"),
    )
    g["broken"] = g["healed"] + g["failed"]
    g = g[g["broken"] > 0].copy()
    g["hsr"] = g["healed"] / g["broken"]
    return g


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 5000,
                     seed: int = 0) -> float:
    """Two-sided permutation test on mean difference."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if diff >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1]."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def magnitude(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def main() -> None:
    df = pd.read_csv(RESULTS_FILE)
    run_df = per_run_hsr(df)

    # --------------------------------------------------------------
    # Bootstrap CI on per-run mean HSR
    # --------------------------------------------------------------
    values = run_df["hsr"].to_numpy()
    mean, lo, hi = bootstrap_ci(values, n_boot=5000)

    # --------------------------------------------------------------
    # Per-version bootstrap CIs
    # --------------------------------------------------------------
    version_rows = []
    for ver, sub in run_df.groupby("ui_version"):
        v = sub["hsr"].to_numpy()
        m, lo_v, hi_v = bootstrap_ci(v, n_boot=5000, seed=abs(hash(ver)) % 2**31)
        version_rows.append({
            "version": ver,
            "n_runs": len(v),
            "mean_hsr": round(m, 4),
            "ci_lo": round(lo_v, 4),
            "ci_hi": round(hi_v, 4),
        })
    version_df = pd.DataFrame(version_rows).sort_values("version")

    # --------------------------------------------------------------
    # Pairwise permutation test + Cliff's delta between versions
    # --------------------------------------------------------------
    versions = sorted(run_df["ui_version"].unique())
    pair_rows = []
    for i in range(len(versions)):
        for j in range(i + 1, len(versions)):
            v1 = run_df[run_df["ui_version"] == versions[i]]["hsr"].to_numpy()
            v2 = run_df[run_df["ui_version"] == versions[j]]["hsr"].to_numpy()
            p = permutation_test(v1, v2, n_perm=5000,
                                 seed=abs(hash((versions[i], versions[j]))) % 2**31)
            d = cliffs_delta(v1, v2)
            pair_rows.append({
                "a": versions[i], "b": versions[j],
                "mean_a": round(float(v1.mean()), 4),
                "mean_b": round(float(v2.mean()), 4),
                "p_perm": round(p, 4),
                "cliffs_delta": round(d, 4),
                "magnitude": magnitude(d),
            })
    pair_df = pd.DataFrame(pair_rows)

    # --------------------------------------------------------------
    # CSV + TXT
    # --------------------------------------------------------------
    out = pd.concat(
        [version_df.assign(scope="per_version"),
         pair_df.assign(scope="pairwise")],
        ignore_index=True, sort=False,
    )
    out.to_csv(REPORT_DIR / "bootstrap_effect_size.csv", index=False)

    lines = []
    lines.append("BOOTSTRAP CI + PERMUTATION + CLIFF'S DELTA")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Per-run HSR (n={len(values)}):")
    lines.append(f"  mean={mean:.4f}, bootstrap 95% CI [{lo:.4f}, {hi:.4f}]")
    lines.append("")
    lines.append("Per-version bootstrap CIs:")
    for _, r in version_df.iterrows():
        lines.append(
            f"  {r['version']:<12} mean={r['mean_hsr']:.4f} "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] (n_runs={r['n_runs']})"
        )
    lines.append("")
    lines.append("Pairwise permutation test + Cliff's delta:")
    for _, r in pair_df.iterrows():
        lines.append(
            f"  {r['a']} vs {r['b']}: "
            f"mean_diff={r['mean_b']-r['mean_a']:+.4f}, "
            f"p={r['p_perm']:.4f}, "
            f"delta={r['cliffs_delta']:+.4f} ({r['magnitude']})"
        )
    (REPORT_DIR / "bootstrap_effect_size.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # --------------------------------------------------------------
    # LaTeX table (pairwise only, per-version is already covered by Wilson CI)
    # --------------------------------------------------------------
    latex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pairwise Permutation Test and Cliff's Delta Effect Size on Per-Run HSR}",
        r"\label{tab:bootstrap_effect_size}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"A & B & Mean(B) - Mean(A) & Perm.\ $p$ & Cliff's $\delta$ (mag.) \\",
        r"\midrule",
    ]
    for _, r in pair_df.iterrows():
        a = r["a"].replace("_", r"\_")
        b = r["b"].replace("_", r"\_")
        latex.append(
            f"{a} & {b} & {r['mean_b']-r['mean_a']:+.3f} & "
            f"{r['p_perm']:.3f} & "
            f"{r['cliffs_delta']:+.3f} ({r['magnitude']}) \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORT_DIR / "bootstrap_effect_size_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    # --------------------------------------------------------------
    # Figure: per-run HSR distribution with bootstrap mean CI
    # --------------------------------------------------------------
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=20, color="#4c78a8", edgecolor="black", alpha=0.8)
    ax.axvline(mean, color="black", linestyle="--",
               label=f"mean = {mean:.3f}")
    ax.axvspan(lo, hi, alpha=0.2, color="#2ca02c",
               label=f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    ax.set_xlabel("Per-run HSR")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Per-Run HSR Distribution (n = {len(values)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_run_hsr_bootstrap.png", dpi=300)
    plt.close()

    print("\n".join(lines))
    print("\nArtifacts written.")


if __name__ == "__main__":
    main()
