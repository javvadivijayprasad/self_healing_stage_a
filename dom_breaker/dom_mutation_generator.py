from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

# -------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app_versions"
REPORT_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORT_DIR / "mutation_report.csv"

VERSIONS = 5
MAX_MUTATIONS_PER_PAGE = 6

# elements worth mutating (UI elements)
TARGET_TAGS = [
    "button",
    "a",
    "input",
    "span",
    "p",
    "h1",
    "h2",
    "div",
]

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def ensure_dirs():

    REPORT_DIR.mkdir(exist_ok=True)

    for i in range(2, VERSIONS + 1):

        v = APP_DIR / f"version_{i}"

        if v.exists():
            shutil.rmtree(v)

        v.mkdir(parents=True)


def log_mutation(mutations, page, mtype, element, old, new):

    mutations.append(
        {
            "page": page,
            "mutation_type": mtype,
            "element": element,
            "old_value": old,
            "new_value": new,
        }
    )


# -------------------------------------------------------
# Mutation functions
# -------------------------------------------------------

def mutate_id(el, page, mutations):

    if el.has_attr("id"):

        old = el["id"]
        new = f"{old}-v"

        el["id"] = new

        log_mutation(mutations, page, "id_change", el.name, old, new)


def mutate_class(el, page, mutations):

    if el.has_attr("class"):

        old = " ".join(el["class"])
        new = "ui-v2"

        el["class"] = [new]

        log_mutation(mutations, page, "class_change", el.name, old, new)


def mutate_text(el, page, mutations):

    if el.string:

        old = el.string.strip()

        replacements = {
            "Login": "Sign In",
            "Add to cart": "Add item",
            "Checkout": "Proceed",
            "Finish Order": "Complete Order",
            "Products": "Catalog",
            "Cart": "Basket",
        }

        if old in replacements:

            new = replacements[old]

            el.string.replace_with(new)

            log_mutation(mutations, page, "text_change", el.name, old, new)


def mutate_placeholder(el, page, mutations):

    if el.has_attr("placeholder"):

        old = el["placeholder"]

        replacements = {
            "Username": "User ID",
            "Password": "Account Password",
            "Postal Code": "ZIP Code",
        }

        if old in replacements:

            new = replacements[old]

            el["placeholder"] = new

            log_mutation(mutations, page, "placeholder_change", el.name, old, new)


def remove_attribute(el, page, mutations):

    if el.has_attr("id"):

        old = el["id"]

        del el["id"]

        log_mutation(mutations, page, "attribute_removed", el.name, old, "removed")


def mutate_dom_wrap(soup, page, mutations):

    products = soup.find_all("div", {"class": "product"})

    for p in products:

        wrapper = soup.new_tag("section")
        wrapper["class"] = "product-wrapper"

        p.wrap(wrapper)

        log_mutation(
            mutations,
            page,
            "dom_wrap",
            "div.product",
            "div.product",
            "section.product-wrapper > div.product",
        )


def mutate_reorder(soup, page, mutations):

    products = soup.find_all("div", {"class": "product"})

    if len(products) > 1:

        parent = products[0].parent

        random.shuffle(products)

        for p in products:
            parent.append(p)

        log_mutation(
            mutations,
            page,
            "element_reorder",
            "div.product",
            "original order",
            "random order",
        )


# -------------------------------------------------------
# Page mutation
# -------------------------------------------------------

def mutate_page(html_path):

    page = html_path.name

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    mutations = []

    elements = [e for e in soup.find_all(TARGET_TAGS)]

    random.shuffle(elements)

    count = 0

    for el in elements:

        if count >= MAX_MUTATIONS_PER_PAGE:
            break

        mutation = random.choice(
            [
                mutate_id,
                mutate_class,
                mutate_text,
                mutate_placeholder,
                remove_attribute,
            ]
        )

        mutation(el, page, mutations)

        count += 1

    mutate_dom_wrap(soup, page, mutations)
    mutate_reorder(soup, page, mutations)

    return soup, mutations


# -------------------------------------------------------
# Version generation
# -------------------------------------------------------

def generate_versions():

    ensure_dirs()

    all_mutations = []

    for v in range(2, VERSIONS + 1):

        src = APP_DIR / f"version_{v-1}"
        dst = APP_DIR / f"version_{v}"

        print(f"Generating version_{v}")

        for file in src.iterdir():

            target = dst / file.name

            if file.suffix != ".html":

                shutil.copy(file, target)
                continue

            soup, mutations = mutate_page(file)

            with open(target, "w", encoding="utf-8") as f:
                f.write(str(soup))

            all_mutations.extend(mutations)

    write_report(all_mutations)


# -------------------------------------------------------
# Report
# -------------------------------------------------------

def write_report(mutations):

    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["page", "mutation_type", "element", "old_value", "new_value"],
        )

        writer.writeheader()
        writer.writerows(mutations)

    print("\nMutation Report Saved:", REPORT_FILE)

    stats = {}

    for m in mutations:
        t = m["mutation_type"]
        stats[t] = stats.get(t, 0) + 1

    print("\nMutation Distribution")
    print("----------------------")

    for k, v in stats.items():
        print(k, ":", v)


# -------------------------------------------------------

if __name__ == "__main__":

    generate_versions()