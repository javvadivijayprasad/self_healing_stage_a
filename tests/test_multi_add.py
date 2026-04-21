from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_multi_add_to_cart_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/inventory.html")
    driver.implicitly_wait(3)

    items = [
        ("add_backpack","add-backpack"),
        ("add_bike","add-bike"),
        ("add_shirt","add-shirt")
    ]

    for name, locator in items:
        el, log = healer.find_element(driver,"multi_add_to_cart_test","inventory",name,"id",locator)
        step_logs.append(log)

        try:
            if el:
                el.click()
        except:
            pass

    cart_link, log4 = healer.find_element(driver,"multi_add_to_cart_test","inventory","shopping_cart_link","id","shopping_cart_link")
    step_logs.append(log4)

    if cart_link:
        cart_link.click()

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "multi_add_to_cart_test", "step_logs": step_logs}