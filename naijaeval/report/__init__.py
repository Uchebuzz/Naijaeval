"""Report generation: JSON and HTML output."""

from naijaeval.report.html_report import save_html, to_html
from naijaeval.report.json_report import save_json, to_json

__all__ = ["to_json", "save_json", "to_html", "save_html"]
