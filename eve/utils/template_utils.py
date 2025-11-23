from __future__ import annotations

import pathlib

from jinja2 import Template


def load_template(filename: str) -> Template:
    """Load and compile a template from the templates directory"""
    TEMPLATE_DIR = pathlib.Path(__file__).parent.parent / "prompt_templates"
    template_path = TEMPLATE_DIR / f"{filename}.txt"
    with open(template_path) as f:
        return Template(f.read())
