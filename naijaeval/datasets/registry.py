"""Dataset registry — auto-populated when loaders.py is imported."""

# Loaders self-register via the @register_dataset decorator.
# Import loaders here so registrations happen at package import time.
from naijaeval.datasets import loaders  # noqa: F401
