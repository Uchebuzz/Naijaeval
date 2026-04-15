"""Report generation: JSON and HTML output."""

from naijaeval.report.json_report import to_json, save_json
from naijaeval.report.html_report import to_html, save_html

__all__ = ["to_json", "save_json", "to_html", "save_html"]
