"""
Generate mutation-vs-healing summary by joining the step-level healing report
against the mutation report on the original locator.

Previous bugs fixed:
  1. The old script used `results["status"].eq("heal_success")`, but
     `results.csv` records TEST-level status (passed/error), not step-level
     healing status, so `healed` was effectively always 0.
  2. The old script assigned the SAME global healed/failed totals to every
     mutation_type row, making per-mutation aggregation meaningless.

This rewrite uses `locator_healing_report.csv` (step-level) as the source of
truth for heal outcomes, left-joins it against `mutation_report.csv` on
`original_locator == old_value` to attach a mutation_type to each broken
locator event, and aggregates healed/failed counts per mutation class.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

HEALING_FILE = BASE_DIR / "reports/locator_healing_report.csv"
MUTATION_FILE = BASE_DIR / "reports/mutation_report.csv"
OUTPUT_FILE = BASE_DIR / "reports/mutation_healing_summary.csv"
LATEX_OUTPUT_FILE = BASE_DIR / "reports/mutation_failure_impact_table.tex"


def main():
    print("Loading step-level healing report...")
    healing = pd.read_csv(HEALING_FILE)

    print("Loading mutation report...")
    mutations = pd.read_csv(MUTATION_FILE)

    # ------------------------------------------------------------------
    # Normalize types
    # ------------------------------------------------------------------
    healing["healing_success"] = (
        pd.to_numeric(healing["healing_success"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Build mutation lookup: old_value -> mutation_type
    # Multiple mutations can share an old_value for different pages; for
    # per-class aggregation we collapse to the first mutation_type per
    # old_value (each old_value in our dataset is unique per type).
    # ------------------------------------------------------------------
    mutation_lookup = (
        mutations.dropna(subset=["old_value"])
        .drop_duplicates(subset=["old_value"])[["old_value", "mutation_type"]]
        .rename(columns={"old_value": "original_locator"})
    )

    # ------------------------------------------------------------------
    # Join healing report to mutation_type
    # ------------------------------------------------------------------
    merged = healing.merge(mutation_lookup, how="left", on="original_locator")

    mapped = merged.dropna(subset=["mutation_type"]).copy()
    unmapped = merged[merged["mutation_type"].isna()]

    # A baseline_success row means the original locator still worked, so it
    # is NOT a healing attempt and must be excluded from attempts/failures
    # aggregation. Without this, baseline_success rows inflate the failure
    # count by roughly one-per-row.
    if "status" in mapped.columns:
        mapped = mapped[mapped["status"] != "baseline_success"].copy()

    print(
        f"Joined rows: {len(merged)} total, "
        f"{len(mapped)} are healing attempts mapped to a mutation_type, "
        f"{len(unmapped)} unmapped (no matching old_value)."
    )

    # ------------------------------------------------------------------
    # Aggregate per mutation_type
    # ------------------------------------------------------------------
    mutation_counts = (
        mutations.groupby("mutation_type").size().rename("mutation_count")
    )

    grp = mapped.groupby("mutation_type")["healing_success"]
    healed = grp.sum().rename("healed")
    attempted = grp.size().rename("attempts")
    failed = (attempted - healed).rename("failures")

    summary = pd.concat([mutation_counts, healed, failed, attempted], axis=1)
    summary = summary.fillna(0).astype({"healed": int, "failures": int, "attempts": int})

    summary["recovery_rate"] = summary.apply(
        lambda r: (r["healed"] / r["attempts"]) if r["attempts"] > 0 else 0.0,
        axis=1,
    ).round(4)

    summary = summary.reset_index().sort_values("mutation_count", ascending=False)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    summary.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)
    print(summary.to_string(index=False))

    # ------------------------------------------------------------------
    # Write IEEE-style LaTeX table (overwrites the old, table that was
    # just a count; this version includes per-class recovery).
    # ------------------------------------------------------------------
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-Mutation-Class Healing Outcomes}",
        r"\label{tab:mutation_failure_impact}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Mutation Type & Injected & Attempts & Healed & Recovery \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        mtype = r["mutation_type"].replace("_", "\\_")
        lines.append(
            f"{mtype} & "
            f"{int(r['mutation_count'])} & "
            f"{int(r['attempts'])} & "
            f"{int(r['healed'])} & "
            f"{r['recovery_rate']*100:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    LATEX_OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("Saved:", LATEX_OUTPUT_FILE)


if __name__ == "__main__":
    main()
