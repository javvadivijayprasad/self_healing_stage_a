from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_cart_page_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/cart.html")
    driver.implicitly_wait(3)

    cart_title, log1 = healer.find_element(driver, "cart_page_test","cart","cart_title","id","cart-title")
    step_logs.append(log1)

    cart_item1, log2 = healer.find_element(driver,"cart_page_test","cart","cart_item1","id","cart-item1")
    step_logs.append(log2)

    cart_item2, log3 = healer.find_element(driver,"cart_page_test","cart","cart_item2","id","cart-item2")
    step_logs.append(log3)

    checkout_link, log4 = healer.find_element(driver,"cart_page_test","cart","checkout_link","id","checkout-link")
    step_logs.append(log4)

    try:
        if checkout_link:
            checkout_link.click()
    except Exception as e:
        step_logs.append({"interaction_error": str(e)})

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "cart_page_test", "step_logs": step_logs}