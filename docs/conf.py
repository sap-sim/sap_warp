"""Sphinx configuration for the SAP Warp documentation."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

with (REPO_ROOT / "pyproject.toml").open("rb") as f:
    PROJECT_INFO = tomllib.load(f)["project"]

project = "SAP Warp"
author = "UCLA AIVC Lab and TRI"
copyright = ""
release = str(PROJECT_INFO.get("version", "0.1.0"))

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

extlinks = {
    "issue": ("https://github.com/sap-sim/sap_warp/issues/%s", "issue #%s"),
    "pr": ("https://github.com/sap-sim/sap_warp/pull/%s", "PR #%s"),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "warp": ("https://nvidia.github.io/warp/stable", None),
}

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
autosummary_imported_members = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "pydata_sphinx_theme"
html_title = "SAP Warp"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_permalinks_icon = "#"
html_show_sourcelink = False

html_theme_options = {
    "navigation_depth": 1,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "collapse_navigation": False,
    "show_prev_next": False,
    "use_edit_page_button": False,
    "logo": {
        "text": f"SAP Warp <span style='font-size: 0.8em; color: #888;'>({release})</span>",
    },
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "index": [],
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sap-sim/sap_warp",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "footer_start": [],
    "footer_end": ["theme-version"],
}

html_sidebars = {"**": ["sidebar-nav-bs.html"], "index": ["sidebar-nav-bs.html"]}

pygments_style = "default"
pygments_dark_style = "monokai"

linkcheck_ignore = [
    r"http://localhost:\d+",
    r"http://0\.0\.0\.0:\d+",
]

# Keep docs builds deterministic in CI when imported modules inspect runtime
# settings. The current pages avoid heavyweight autodoc imports, but future API
# pages can share this switch.
os.environ.setdefault("SAP_WARP_DOCS_BUILD", "1")
