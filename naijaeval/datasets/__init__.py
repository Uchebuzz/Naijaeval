"""Datasets subpackage — imports loaders to trigger self-registration."""

from naijaeval.datasets import loaders  # noqa: F401 — triggers @register_dataset calls

__all__ = ["loaders"]
