from __future__ import annotations
from typing import Any, Dict, List, Tuple


def run_login_test(driver: Any, healer: Any, base_root_url: str) -> Tuple[bool, Dict[str, Any]]:

    step_logs: List[Dict[str, Any]] = []

    driver.get(f"{base_root_url}/login.html")
    driver.implicitly_wait(3)

    username, log1 = healer.find_element(
        driver=driver,
        test_case="login_test",
        page_name="login",
        element_name="username_input",
        locator_type="id",
        locator_value="username-input",
    )
    step_logs.append(log1)

    password, log2 = healer.find_element(
        driver=driver,
        test_case="login_test",
        page_name="login",
        element_name="password_input",
        locator_type="id",
        locator_value="password-input",
    )
    step_logs.append(log2)

    login_button, log3 = healer.find_element(
        driver=driver,
        test_case="login_test",
        page_name="login",
        element_name="login_button",
        locator_type="id",
        locator_value="login-button",
    )
    step_logs.append(log3)

    try:
        if username:
            username.clear()
            username.send_keys("standard_user")

        if password:
            password.clear()
            password.send_keys("secret_sauce")

        if login_button:
            login_button.click()

    except Exception as e:
        step_logs.append({"interaction_error": str(e)})

    success = all(log.get("found", True) for log in step_logs)

    return success, {"test_name": "login_test", "step_logs": step_logs}