import pandas as pd
import json
import ast
import re
from pathlib import Path

# Pattern used by the healer when raising after a failed recovery attempt:
#   "... for <test_case>/<element_name>."
ERROR_ELEMENT_PATTERN = re.compile(r"for ([a-zA-Z_0-9]+)/([a-zA-Z_0-9]+)")


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_FILE = BASE_DIR / "reports" / "results.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "locator_healing_report.csv"


def parse_details(details):

    if pd.isna(details):
        return None

    try:
        return json.loads(details)
    except Exception:
        try:
            return ast.literal_eval(details)
        except Exception:
            return None


def generate_report():

    print("Reading:", RESULTS_FILE)

    df = pd.read_csv(RESULTS_FILE)

    rows = []

    for _, r in df.iterrows():

        details_raw = r.get("details")
        details = parse_details(details_raw)

        # ----------------------------------------------------------------
        # Error rows carry a traceback string instead of structured step
        # logs. Parse out test_case/element_name so failed heals are still
        # attributed to a locator. Without this branch, the 850 failed
        # healings are silently dropped from the step-level report.
        # ----------------------------------------------------------------
        if not details:
            if isinstance(details_raw, str):
                m = ERROR_ELEMENT_PATTERN.search(details_raw)
                if m:
                    test_name_err, element_name_err = m.group(1), m.group(2)
                    rows.append({
                        "ui_version": r.get("ui_version"),
                        "test_name": test_name_err,
                        "element_name": element_name_err,
                        # Convention across the repo: locator strings use
                        # hyphens while Python identifiers use underscores.
                        "original_locator": element_name_err.replace("_", "-"),
                        "healed_locator": "",
                        "healing_score": 0,
                        "healing_success": 0,
                        "healing_failed": 1,
                        "status": "heal_failed",
                    })
            continue

        test_name = details.get("test_name", r.get("test_name"))
        ui_version = r.get("ui_version")

        steps = details.get("step_logs", [])

        for step in steps:

            rows.append({

                "ui_version": ui_version,

                "test_name": test_name,

                "element_name": step.get("element_name", "unknown"),

                "original_locator": step.get("original_locator", ""),

                "healed_locator": step.get("healed_locator", ""),

                "healing_score": step.get("healing_score", 0),

                "healing_success": 1 if step.get("healed", False) else 0,

                # Bug fix: step logs carry outcome in `status`
                # (heal_success / heal_failed / baseline_success), not a
                # `failed` boolean. Mark healing_failed=1 when the framework
                # attempted a heal but could not resolve the candidate live.
                "healing_failed": 1 if step.get("status") == "heal_failed" else 0,

                "status": step.get("status", ""),

            })

    report = pd.DataFrame(rows)

    if report.empty:

        print("WARNING: No locator healing events found.")
        return

    report.to_csv(OUTPUT_FILE, index=False)

    print("locator_healing_report.csv generated with", len(report), "rows")


if __name__ == "__main__":
    generate_report()