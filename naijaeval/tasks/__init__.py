"""Tasks subpackage."""

from naijaeval.tasks.asr import ASRTask
from naijaeval.tasks.summarisation import SummarisationTask
from naijaeval.tasks.translation import TranslationTask

__all__ = ["TranslationTask", "ASRTask", "SummarisationTask"]
