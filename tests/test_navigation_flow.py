from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_navigation_flow_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/index.html")
    driver.implicitly_wait(3)

    nav_login, log1 = healer.find_element(driver,"navigation_flow_test","index","nav_login","id","nav-login")
    step_logs.append(log1)

    if nav_login:
        nav_login.click()

    products, log2 = healer.find_element(driver,"navigation_flow_test","login","login_products_link","id","login-products-link")
    step_logs.append(log2)

    if products:
        products.click()

    cart, log3 = healer.find_element(driver,"navigation_flow_test","inventory","shopping_cart_link","id","shopping_cart_link")
    step_logs.append(log3)

    if cart:
        cart.click()

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "navigation_flow_test", "step_logs": step_logs}