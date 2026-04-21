from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from healing.locator_store import LocatorStore
from healing.self_heal_engine import SelfHealEngine
from healing.ranker_factory import create_ranker, VALID_MODES
from healing.utils import ensure_dir, env_bool, timestamp

from tests.test_inventory import run_add_to_cart_test
from tests.test_login import run_login_test
from tests.test_home_navigation import run_home_navigation_test
from tests.test_cart import run_cart_page_test
from tests.test_checkout import run_checkout_form_test
from tests.test_products import run_products_listing_test
from tests.test_multi_add import run_multi_add_to_cart_test
from tests.test_navigation_flow import run_navigation_flow_test
import random
import time

BASELINE_ROOT_URL = "http://127.0.0.1:8000/version_1"
RESULTS_FILE = Path("reports/results.csv")

# Set by CLI / main(); consumed by run_suite() when instantiating the healer.
HEALER_MODE = "heuristic"

_SEED_COUNTER = [0]


def run_seed_counter() -> int:
    _SEED_COUNTER[0] += 1
    return _SEED_COUNTER[0]


# -----------------------------------------------------
# Driver
# -----------------------------------------------------

def build_driver(headless: bool = False) -> webdriver.Chrome:

    options = ChromeOptions()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = ChromeService(ChromeDriverManager().install())

    return webdriver.Chrome(service=service, options=options)


# -----------------------------------------------------
# Bootstrap locator metadata
# -----------------------------------------------------

def _preflight_baseline_check() -> None:
    """Fail fast with a helpful message if the baseline URL isn't reachable
    or doesn't contain the expected fixture."""
    import urllib.request
    import urllib.error

    probes = [
        (f"{BASELINE_ROOT_URL}/login.html",     "login-button"),
        (f"{BASELINE_ROOT_URL}/inventory.html", "add-backpack"),
    ]
    for url, needle in probes:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"[PREFLIGHT] Cannot reach {url}: {exc}.\n"
                f"Did you start the local web server in a separate terminal?\n"
                f"    python -m http.server 8000 --directory app_versions"
            )
        if needle not in body:
            raise SystemExit(
                f"[PREFLIGHT] {url} responded but does not contain '{needle}'.\n"
                f"The baseline fixture under app_versions/version_1/ may be "
                f"missing or stale. Re-check the folder, or restart the "
                f"http.server bound to --directory app_versions."
            )
    print("[PREFLIGHT] baseline URL reachable and fixtures present.")


def bootstrap_locator_metadata(locator_store: LocatorStore):

    print("[BOOTSTRAP] Collecting baseline locator metadata")
    _preflight_baseline_check()

    driver = build_driver(headless=True)

    healer = SelfHealEngine(locator_store=locator_store, bootstrap_mode=True)

    try:

        tests = [
            run_login_test,
            run_add_to_cart_test,
            run_home_navigation_test,
            run_cart_page_test,
            run_checkout_form_test,
            run_products_listing_test,
            run_multi_add_to_cart_test,
            run_navigation_flow_test,
        ]

        for fn in tests:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    fn(driver, healer, BASELINE_ROOT_URL)
                    driver.delete_all_cookies()
                    break
                except Exception as exc:
                    driver.delete_all_cookies()
                    if attempt < max_retries:
                        print(f"[BOOTSTRAP] {fn.__name__} failed (attempt {attempt}/{max_retries}): {exc}")
                        print(f"[BOOTSTRAP] Retrying in 3 seconds...")
                        time.sleep(3)
                    else:
                        print(f"[BOOTSTRAP] {fn.__name__} failed after {max_retries} attempts: {exc}")
                        raise

    finally:

        driver.quit()

    print("[BOOTSTRAP] Completed")


# -----------------------------------------------------
# Step log summary
# -----------------------------------------------------

def summarize_step_logs(details: Dict[str, Any]) -> Dict[str, int]:

    logs = details.get("step_logs", [])

    healed = sum(1 for x in logs if x.get("healed", False))

    baseline = sum(
        1 for x in logs
        if not x.get("healed", False) and not x.get("failed", False)
    )

    failed = sum(1 for x in logs if x.get("failed", False))

    return {
        "healed": healed,
        "baseline": baseline,
        "failed": failed,
    }


# -----------------------------------------------------
# Execute one UI version
# -----------------------------------------------------

def run_suite(locator_store: LocatorStore, base_root_url: str, ui_version: str):

    driver = build_driver(headless=env_bool("HEADLESS", False))

    ranker = create_ranker(HEALER_MODE, threshold=0.45, seed=run_seed_counter())
    healer = SelfHealEngine(locator_store=locator_store, ranker=ranker)

    run_id = timestamp()

    rows: List[Dict[str, Any]] = []

    tests = [
        ("login_test", run_login_test),
        ("inventory_test", run_add_to_cart_test),
        ("home_navigation_test", run_home_navigation_test),
        ("cart_page_test", run_cart_page_test),
        ("checkout_form_test", run_checkout_form_test),
        ("products_listing_test", run_products_listing_test),
        ("multi_add_to_cart_test", run_multi_add_to_cart_test),
        ("navigation_flow_test", run_navigation_flow_test),
    ]
    random.shuffle(tests)
    try:

        for name, fn in tests:

            print(f"[RUN] {ui_version} → {name}")

            try:

                success, details = fn(driver, healer, base_root_url)

                counts = summarize_step_logs(details)

                rows.append({

                    "run_id": run_id,
                    "ui_version": ui_version,
                    "healer_mode": HEALER_MODE,
                    "test_name": name,
                    "status": "passed" if success else "failed",
                    "success": success,
                    "healed_steps": counts["healed"],
                    "baseline_steps": counts["baseline"],
                    "failed_steps": counts["failed"],
                    "details": json.dumps(details)

                })

            except Exception as exc:

                rows.append({

                    "run_id": run_id,
                    "ui_version": ui_version,
                    "healer_mode": HEALER_MODE,
                    "test_name": name,
                    "status": "error",
                    "success": False,
                    "healed_steps": 0,
                    "baseline_steps": 0,
                    "failed_steps": 1,
                    "details": f"{exc}\n{traceback.format_exc()}"

                })

            finally:

                driver.delete_all_cookies()

        return rows

    finally:

        # Flush heal event logger before closing driver
        try:
            healer.flush_event_log()
        except Exception:
            pass

        driver.quit()


# -----------------------------------------------------
# Write results
# -----------------------------------------------------

def write_results(rows: List[Dict[str, Any]], results_file_override: str = None):

    if not rows:
        print("No results to write.")
        return

    ensure_dir(RESULTS_FILE.parent)

    if results_file_override:
        # Explicit output path provided (e.g. for multi-model comparison)
        targets = [Path(results_file_override)]
    elif HEALER_MODE == "heuristic":
        # The canonical results.csv holds the heuristic mode's data because the
        # rest of the analysis pipeline reads that file. Non-heuristic modes
        # write ONLY to a mode-tagged copy so the ablation sweep does not
        # clobber the primary results.
        targets = [RESULTS_FILE]
    else:
        targets = [RESULTS_FILE.parent / f"results_{HEALER_MODE}.csv"]

    for out_path in targets:
        ensure_dir(out_path.parent)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("[RESULTS] saved to", out_path)


# -----------------------------------------------------
# Main experiment runner
# -----------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-healing locator experiment."
    )
    parser.add_argument(
        "--healer-mode",
        choices=list(VALID_MODES),
        default=os.environ.get("HEALER_MODE", "heuristic"),
        help="Ranker strategy to use when the baseline locator fails "
             "(default: heuristic). Use rule_based/random/none for ablation.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=int(os.environ.get("EXPERIMENT_RUNS", "50")),
        help="Number of experiment runs (default: 50).",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Override the output CSV path (e.g. reports/results_ml_xgb.csv). "
             "If not set, uses the default naming convention.",
    )
    return parser.parse_args()


def main():

    global HEALER_MODE

    args = _parse_args()
    HEALER_MODE = args.healer_mode

    print(f"[CONFIG] healer_mode={HEALER_MODE}  runs={args.runs}")

    ensure_dir("reports")

    locator_store = LocatorStore()

    bootstrap_locator_metadata(locator_store)

    versions = [
        "version_2",
        "version_3",
        "version_4",
        "version_5",
    ]

    experiment_runs = args.runs

    all_rows = []

    for run in range(experiment_runs):
        seed = int(time.time())
        random.seed(seed)

        print("\n==============================")
        print("EXPERIMENT RUN", run + 1)
        print("==============================")

        for version in versions:

            url = f"http://127.0.0.1:8000/{version}"

            rows = run_suite(locator_store, url, version)

            all_rows.extend(rows)

    write_results(all_rows, results_file_override=args.results_file)


if __name__ == "__main__":

    main()