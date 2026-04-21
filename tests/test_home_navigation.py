from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_home_navigation_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/index.html")
    driver.implicitly_wait(3)

    elements = [
        ("nav_login","nav-login"),
        ("nav_products","nav-products"),
        ("nav_cart","nav-cart"),
        ("welcome_text","welcome-text")
    ]

    for name, locator in elements:
        el, log = healer.find_element(driver,"home_navigation_test","index",name,"id",locator)
        step_logs.append(log)

        try:
            if el:
                _ = el.text
        except:
            pass

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "home_navigation_test", "step_logs": step_logs}