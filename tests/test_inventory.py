from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_add_to_cart_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/inventory.html")
    driver.implicitly_wait(5)
    import time
    time.sleep(1)  # Allow page to fully render

    add_backpack, log1 = healer.find_element(
        driver,
        "inventory_test",
        "inventory",
        "add_backpack_button",
        "id",
        "add-backpack"
    )
    step_logs.append(log1)

    cart_link, log2 = healer.find_element(
        driver,
        "inventory_test",
        "inventory",
        "shopping_cart_link",
        "id",
        "shopping_cart_link"
    )
    step_logs.append(log2)

    try:
        if add_backpack:
            add_backpack.click()

        if cart_link:
            cart_link.click()

    except Exception as e:
        step_logs.append({
            "failed": True,
            "interaction_error": str(e)
        })

    success = all(not log.get("failed", False) for log in step_logs)

    return success, {
        "test_name": "inventory_test",
        "step_logs": step_logs
    }