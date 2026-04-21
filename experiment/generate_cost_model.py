"""Cost-of-maintenance model.

Translates the healing outcomes into an engineering-time saving estimate
under published per-locator repair times. The purpose is to ground the
healing-success-rate numbers in a business-relevant metric so reviewers
and practitioners can compare the framework's payoff against its
implementation cost.

Assumptions are conservative and documented inline. The per-locator
repair time range (10-20 min) is in the range reported by
Hammoudi et al. 2016 and Stocco et al. 2018 for manual locator repair
in record/replay and programmable Selenium tests.

Outputs:
  - reports/cost_model.csv
  - reports/cost_model_table.tex
  - reports/cost_model.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"
RESULTS_FILE = REPORT_DIR / "results.csv"


# Published per-locator-repair time estimates (minutes)
REPAIR_TIME_LO_MIN = 10
REPAIR_TIME_HI_MIN = 20


def main() -> None:
    df = pd.read_csv(RESULTS_FILE)

    healed = int(df["healed_steps"].sum())
    failed = int(df["failed_steps"].sum())
    broken = healed + failed

    rows = []
    for t_min in (REPAIR_TIME_LO_MIN, REPAIR_TIME_HI_MIN):
        # Avoided engineering time = healed events * repair time
        avoided_hours = healed * t_min / 60.0
        # Remaining time still required for unresolved failures
        residual_hours = failed * t_min / 60.0
        total_without_framework = broken * t_min / 60.0
        pct_avoided = avoided_hours / total_without_framework if total_without_framework else 0.0
        rows.append({
            "per_locator_min": t_min,
            "broken_events": broken,
            "healed_events": healed,
            "failed_events": failed,
            "avoided_engineer_hours": round(avoided_hours, 1),
            "residual_engineer_hours": round(residual_hours, 1),
            "total_without_framework_hours": round(total_without_framework, 1),
            "fraction_avoided": round(pct_avoided, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "cost_model.csv", index=False)
    print("Cost model:")
    print(out.to_string(index=False))

    # Plain-text summary
    lines = [
        "MAINTENANCE COST MODEL",
        "=" * 60,
        f"broken events (healing opportunities): {broken}",
        f"healed: {healed}    unresolved: {failed}",
        "",
        "Per-locator repair time assumptions follow published estimates:",
        "  10-20 min per locator (Hammoudi 2016; Stocco 2018).",
        "",
    ]
    for r in rows:
        lines.append(
            f"At {r['per_locator_min']} min/locator: "
            f"{r['avoided_engineer_hours']:.1f} engineer-hours avoided, "
            f"{r['residual_engineer_hours']:.1f} residual hours, "
            f"{r['fraction_avoided']*100:.1f}% of total maintenance cost eliminated."
        )
    (REPORT_DIR / "cost_model.txt").write_text("\n".join(lines), encoding="utf-8")

    # LaTeX table
    latex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Maintenance Cost Model: Engineer-Hours Avoided by Automated Healing. "
        r"Repair-time assumptions follow Hammoudi et al.\ 2016 and Stocco et al.\ 2018.}",
        r"\label{tab:cost_model}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Repair time & Broken & Healed & Avoided hrs & Residual hrs & \% avoided \\",
        r"\midrule",
    ]
    for r in rows:
        latex.append(
            f"{int(r['per_locator_min'])} min & "
            f"{int(r['broken_events'])} & "
            f"{int(r['healed_events'])} & "
            f"{r['avoided_engineer_hours']:.1f} & "
            f"{r['residual_engineer_hours']:.1f} & "
            f"{r['fraction_avoided']*100:.1f}\\% \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (REPORT_DIR / "cost_model_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )

    print("\nArtifacts written.")


if __name__ == "__main__":
    main()
