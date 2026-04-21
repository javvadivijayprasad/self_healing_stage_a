"""Mutation realism audit.

Maps each of the seven mutation classes produced by the DOM mutation
generator to change patterns documented in prior empirical studies of
real web-application UI evolution. The table is static (no dependency on
runtime CSVs) but is emitted as a .tex file so it stays in sync with the
rest of the pipeline and gets regenerated on every full-pipeline run.

Outputs:
  - reports/mutation_realism_table.tex
  - reports/mutation_realism.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports"


# Each row maps one of our mutation classes to the real-world change
# category documented in the literature. Citation keys match the
# thebibliography entries in the paper source.
MAPPING = [
    {
        "mutation_class": "id_change",
        "real_world_pattern": "Identifier renaming during refactor (e.g. kebab- to camel-case)",
        "reference": "hammoudi2016; stocco2018",
    },
    {
        "mutation_class": "attribute_removed",
        "real_world_pattern": "Inline attribute dropped when styling migrates to CSS classes",
        "reference": "hammoudi2016",
    },
    {
        "mutation_class": "dom_wrap",
        "real_world_pattern": "Component re-parenting (wrapping in a container div)",
        "reference": "yandrapally2014; ricca2019",
    },
    {
        "mutation_class": "element_reorder",
        "real_world_pattern": "Sibling reordering driven by responsive-layout redesigns",
        "reference": "yandrapally2014",
    },
    {
        "mutation_class": "class_change",
        "real_world_pattern": "CSS-class renaming when design system version changes",
        "reference": "hammoudi2016; stocco2018",
    },
    {
        "mutation_class": "text_change",
        "real_world_pattern": "Copy edits and i18n label updates",
        "reference": "ricca2019",
    },
    {
        "mutation_class": "placeholder_change",
        "real_world_pattern": "Form placeholder rewording during UX polish passes",
        "reference": "hammoudi2016",
    },
]


def main() -> None:
    df = pd.DataFrame(MAPPING)
    df.to_csv(REPORT_DIR / "mutation_realism.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Mutation Realism Audit: Injected Mutation Classes Mapped to "
        r"Real-World UI Change Patterns Documented in Prior Work}",
        r"\label{tab:mutation_realism}",
        r"\begin{tabular}{p{0.23\columnwidth}p{0.50\columnwidth}p{0.18\columnwidth}}",
        r"\toprule",
        r"Mutation class & Real-world change pattern & Reference \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        mclass = r["mutation_class"].replace("_", r"\_")
        pattern = r["real_world_pattern"]
        refs = ", ".join(
            f"\\cite{{{k.strip()}}}" for k in r["reference"].split(";")
        )
        lines.append(f"{mclass} & {pattern} & {refs} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (REPORT_DIR / "mutation_realism_table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Mutation realism audit written.")


if __name__ == "__main__":
    main()
