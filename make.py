#!/usr/bin/env python3

import shutil
from pathlib import Path

import pdoc

here = Path(__file__).parent

modules = [
    "bat.attacks",
    "bat.apis",
]

# Render pdoc's documentation into docs/api...
pdoc.render.configure(
    template_directory = here / "pdoc-dark-mode", 
    footer_text="Black-box Adversarial Toolbox",
    search=True,
    logo="https://bat.wuhanstudio.uk/images/bat_dark.png",
    logo_link="https://github.com/wuhanstudio/blackbox-adversarial-toolbox")

pdoc.pdoc(*modules, output_directory = here / "docs")
