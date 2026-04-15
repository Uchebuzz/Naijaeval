"""Tasks subpackage."""

from naijaeval.tasks.translation import TranslationTask
from naijaeval.tasks.asr import ASRTask
from naijaeval.tasks.summarisation import SummarisationTask

__all__ = ["TranslationTask", "ASRTask", "SummarisationTask"]
