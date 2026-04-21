from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_products_listing_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/inventory.html")
    driver.implicitly_wait(3)

    elements = [
        ("inventory_title","inventory-title"),
        ("product_backpack","product-backpack"),
        ("product_bike","product-bike"),
        ("product_shirt","product-shirt")
    ]

    for name, locator in elements:
        el, log = healer.find_element(driver,"products_listing_test","inventory",name,"id",locator)
        step_logs.append(log)

        try:
            if el:
                _ = el.text
        except:
            pass

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "products_listing_test", "step_logs": step_logs}