"""
Statistical analysis of healing-success-rate (HSR) across versions, tests,
and individual experiment runs.

Outputs:
  - reports/statistical_analysis.csv   (machine-readable per-group stats)
  - reports/statistical_analysis.txt   (human-readable report)
  - reports/statistical_analysis_table.tex (IEEE-ready table)
  - figures/hsr_per_version_ci.png     (per-version HSR with 95% CIs)
  - figures/hsr_per_test_ci.png        (per-test HSR with 95% CIs)

Methods:
  * Wilson score 95% confidence intervals for binomial proportions.
  * Chi-square test of independence between ui_version and step outcome.
  * Kruskal-Wallis H test on per-run HSR across versions.
  * Friedman test on per-run HSR across versions, blocked by run_id.
  * Bootstrap standard error of aggregate HSR (1000 resamples).
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Lightweight inline statistics (scipy is not available in this env).
# Chi-square / Kruskal / Friedman p-values are computed via the
# Wilson-Hilferty transformation of the chi-square distribution, which
# is accurate to 3+ decimal places for dof >= 2 and is widely used in
# software-engineering empirical studies that avoid heavy dependencies.
# --------------------------------------------------------------------
def chi2_sf(x: float, dof: int) -> float:
    """Approximate chi-square survival function via Wilson-Hilferty."""
    if dof <= 0 or x <= 0:
        return 1.0
    z = ((x / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(
        2.0 / (9.0 * dof)
    )
    # Upper-tail normal via erfc
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_contingency(table: np.ndarray):
    """Chi-square test of independence for an r x c contingency table."""
    table = np.asarray(table, dtype=float)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return 0.0, 1.0, 0
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    chi2 = float(terms.sum())
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return chi2, chi2_sf(chi2, dof), dof


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank implementation matching scipy's default."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Handle ties: assign mean rank within each tie group
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            mean_rank = (ranks[order[i]] + ranks[order[j - 1]]) / 2.0
            for k in range(i, j):
                ranks[order[k]] = mean_rank
        i = j
    return ranks


def kruskal(*groups):
    """Kruskal-Wallis H test on >=2 groups of values."""
    all_vals = np.concatenate(groups)
    n = len(all_vals)
    k = len(groups)
    if n == 0 or k < 2:
        return float("nan"), float("nan")
    ranks = _rankdata(all_vals)
    idx = 0
    rank_sums = []
    sizes = []
    for g in groups:
        sz = len(g)
        rank_sums.append(ranks[idx:idx + sz].sum())
        sizes.append(sz)
        idx += sz
    h = (12.0 / (n * (n + 1))) * sum(
        (rs ** 2) / sz for rs, sz in zip(rank_sums, sizes)
    ) - 3.0 * (n + 1)
    return float(h), chi2_sf(h, k - 1)


def friedmanchisquare(*blocks):
    """Friedman test; each argument is a column of per-block measurements."""
    arr = np.column_stack([np.asarray(b, dtype=float) for b in blocks])
    n, k = arr.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    # Rank within each row
    ranks = np.zeros_like(arr)
    for i in range(n):
        ranks[i] = _rankdata(arr[i])
    rbar = ranks.mean(axis=0)
    q = 12.0 * n / (k * (k + 1)) * float(np.sum((rbar - (k + 1) / 2.0) ** 2))
    return float(q), chi2_sf(q, k - 1)

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = BASE_DIR / "figures"
RESULTS_FILE = REPORT_DIR / "results.csv"


def wilson_ci(successes: int, trials: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_se(successes: int, trials: int, n_boot: int = 1000, seed: int = 0):
    """Bootstrap SE of an aggregate binomial proportion."""
    if trials == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    draws = rng.binomial(trials, successes / trials, size=n_boot) / trials
    return float(draws.std(ddof=1))


def per_run_hsr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HSR per (run_id, ui_version), skipping runs with 0 broken."""
    g = df.groupby(["run_id", "ui_version"], as_index=False).agg(
        healed=("healed_steps", "sum"),
        failed=("failed_steps", "sum"),
    )
    g["broken"] = g["healed"] + g["failed"]
    g = g[g["broken"] > 0].copy()
    g["hsr"] = g["healed"] / g["broken"]
    return g


def main():
    df = pd.read_csv(RESULTS_FILE)

    # ------------------------------------------------------------------
    # Aggregate HSR with Wilson CI and bootstrap SE
    # ------------------------------------------------------------------
    total_healed = int(df["healed_steps"].sum())
    total_failed = int(df["failed_steps"].sum())
    total_broken = total_healed + total_failed
    agg_p, agg_lo, agg_hi = wilson_ci(total_healed, total_broken)
    agg_se = bootstrap_se(total_healed, total_broken)

    # ------------------------------------------------------------------
    # Per-version HSR
    # ------------------------------------------------------------------
    version_rows = []
    for ver, sub in df.groupby("ui_version"):
        healed = int(sub["healed_steps"].sum())
        failed = int(sub["failed_steps"].sum())
        broken = healed + failed
        p, lo, hi = wilson_ci(healed, broken)
        version_rows.append({
            "group": "version",
            "name": ver,
            "healed": healed,
            "failed": failed,
            "broken": broken,
            "hsr": round(p, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })
    version_df = pd.DataFrame(version_rows)

    # ------------------------------------------------------------------
    # Per-test HSR
    # ------------------------------------------------------------------
    test_rows = []
    for tname, sub in df.groupby("test_name"):
        healed = int(sub["healed_steps"].sum())
        failed = int(sub["failed_steps"].sum())
        broken = healed + failed
        if broken == 0:
            continue
        p, lo, hi = wilson_ci(healed, broken)
        test_rows.append({
            "group": "test",
            "name": tname,
            "healed": healed,
            "failed": failed,
            "broken": broken,
            "hsr": round(p, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
        })
    test_df = pd.DataFrame(test_rows)

    # ------------------------------------------------------------------
    # Chi-square test: ui_version vs outcome (healed vs failed)
    # ------------------------------------------------------------------
    contingency = version_df[["healed", "failed"]].to_numpy()
    chi2, chi2_p, chi2_dof = chi2_contingency(contingency)

    # ------------------------------------------------------------------
    # Kruskal-Wallis on per-run HSR across versions
    # ------------------------------------------------------------------
    run_df = per_run_hsr(df)
    groups = [sub["hsr"].to_numpy() for _, sub in run_df.groupby("ui_version")]
    kw_stat, kw_p = kruskal(*groups) if len(groups) >= 2 else (float("nan"), float("nan"))

    # Friedman requires balanced data across run_id, so pivot and drop NaN
    friedman_stat, friedman_p = float("nan"), float("nan")
    try:
        pivot = run_df.pivot_table(index="run_id", columns="ui_version", values="hsr").dropna()
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
            friedman_stat, friedman_p = friedmanchisquare(
                *[pivot[c].to_numpy() for c in pivot.columns]
            )
    except Exception:
        pass

    # Variance of per-run HSR overall
    run_hsr_mean = float(run_df["hsr"].mean())
    run_hsr_std = float(run_df["hsr"].std(ddof=1))
    run_hsr_n = int(len(run_df))

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    out_df = pd.concat([version_df, test_df], ignore_index=True)
    out_df.to_csv(REPORT_DIR / "statistical_analysis.csv", index=False)

    # ------------------------------------------------------------------
    # Write human-readable report
    # ------------------------------------------------------------------
    lines = []
    lines.append("STATISTICAL ANALYSIS OF HEALING SUCCESS RATE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Aggregate HSR: {agg_p:.4f} ({total_healed}/{total_broken})")
    lines.append(f"  Wilson 95% CI: [{agg_lo:.4f}, {agg_hi:.4f}]")
    lines.append(f"  Bootstrap SE (n=1000): {agg_se:.4f}")
    lines.append("")
    lines.append("Per-version HSR (Wilson 95% CI):")
    for _, r in version_df.iterrows():
        lines.append(
            f"  {r['name']:<12} HSR={r['hsr']:.4f} "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
            f"(n={r['broken']})"
        )
    lines.append("")
    lines.append("Per-test HSR (Wilson 95% CI):")
    for _, r in test_df.iterrows():
        lines.append(
            f"  {r['name']:<24} HSR={r['hsr']:.4f} "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
            f"(n={r['broken']})"
        )
    lines.append("")
    lines.append("Chi-square test of independence (ui_version vs outcome):")
    lines.append(f"  chi2={chi2:.4f}, dof={chi2_dof}, p={chi2_p:.4g}")
    lines.append("")
    lines.append("Kruskal-Wallis test on per-run HSR across versions:")
    lines.append(f"  H={kw_stat:.4f}, p={kw_p:.4g}, groups={len(groups)}")
    lines.append("")
    lines.append("Friedman test on per-run HSR (blocked by run_id):")
    lines.append(f"  chi2={friedman_stat:.4f}, p={friedman_p:.4g}")
    lines.append("")
    lines.append("Per-run HSR distribution:")
    lines.append(f"  mean={run_hsr_mean:.4f}, sd={run_hsr_std:.4f}, n_runs={run_hsr_n}")

    (REPORT_DIR / "statistical_analysis.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Per-Version and Per-Test Healing Success Rate with Wilson 95\% Confidence Intervals}")
    latex.append(r"\label{tab:statistical_analysis}")
    latex.append(r"\begin{tabular}{llccc}")
    latex.append(r"\toprule")
    latex.append(r"Scope & Name & HSR & 95\% CI & n \\")
    latex.append(r"\midrule")
    for _, r in version_df.iterrows():
        nm = r["name"].replace("_", r"\_")
        latex.append(
            f"Version & {nm} & {r['hsr']*100:.1f}\\% & "
            f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] & "
            f"{int(r['broken'])} \\\\"
        )
    latex.append(r"\midrule")
    for _, r in test_df.iterrows():
        nm = r["name"].replace("_", r"\_")
        latex.append(
            f"Test & {nm} & {r['hsr']*100:.1f}\\% & "
            f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] & "
            f"{int(r['broken'])} \\\\"
        )
    latex.append(r"\midrule")
    latex.append(
        f"Aggregate & all & {agg_p*100:.1f}\\% & "
        f"[{agg_lo*100:.1f}, {agg_hi*100:.1f}] & "
        f"{total_broken} \\\\"
    )
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    (REPORT_DIR / "statistical_analysis_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Figures: per-version and per-test HSR with CIs
    # ------------------------------------------------------------------
    def barplot_with_ci(df_in, fname, title):
        x = np.arange(len(df_in))
        p = df_in["hsr"].to_numpy()
        lo = df_in["ci_lo"].to_numpy()
        hi = df_in["ci_hi"].to_numpy()
        err = np.vstack([p - lo, hi - p])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, p, yerr=err, capsize=4, color="#4c78a8", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(df_in["name"], rotation=30, ha="right")
        ax.set_ylabel("Healing Success Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=300)
        plt.close()

    barplot_with_ci(version_df, "hsr_per_version_ci.png",
                    "Per-Version HSR with Wilson 95% CI")
    barplot_with_ci(test_df.sort_values("hsr", ascending=False),
                    "hsr_per_test_ci.png",
                    "Per-Test HSR with Wilson 95% CI")

    print("\n".join(lines))
    print("\nArtifacts written to reports/ and figures/.")


if __name__ == "__main__":
    main()
