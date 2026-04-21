from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_checkout_form_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/checkout.html")
    driver.implicitly_wait(3)

    first_name, log1 = healer.find_element(driver,"checkout_form_test","checkout","first_name","id","first-name")
    step_logs.append(log1)

    last_name, log2 = healer.find_element(driver,"checkout_form_test","checkout","last_name","id","last-name")
    step_logs.append(log2)

    postal_code, log3 = healer.find_element(driver,"checkout_form_test","checkout","postal_code","id","postal-code")
    step_logs.append(log3)

    finish_button, log4 = healer.find_element(driver,"checkout_form_test","checkout","finish_button","id","finish-button")
    step_logs.append(log4)

    try:
        if first_name:
            first_name.send_keys("Vijay")

        if last_name:
            last_name.send_keys("Javvadi")

        if postal_code:
            postal_code.send_keys("07302")

        if finish_button:
            finish_button.click()

    except Exception as e:
        step_logs.append({"interaction_error": str(e)})

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "checkout_form_test", "step_logs": step_logs}