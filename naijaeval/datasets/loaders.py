"""Dataset loaders for NaijaEval benchmarks.

Each loader is registered under a short name via ``@register_dataset``.
Loaders return an iterable of dicts with at minimum ``"source"`` and
``"target"`` keys (plus optional ``"dialect"``, ``"domain"``, etc.).

All loaders rely on the HuggingFace ``datasets`` library.  If a dataset
requires authentication (private datasets), the user must call
``huggingface_hub.login()`` before loading.

Registered datasets
-------------------
- ``menyo20k`` — MENYO-20k: Yoruba-English MT (Adelani et al., 2021)
- ``fleurs_yo`` — FLEURS Yoruba (speech, ASR)
- ``fleurs_ha`` — FLEURS Hausa (speech, ASR)
- ``fleurs_sw`` — FLEURS Swahili (speech, ASR)
- ``naija_mt_sample`` — Built-in 50-sentence sample for quick demos
"""

from __future__ import annotations

from typing import Any

from naijaeval.registry import register_dataset


@register_dataset("menyo20k")
def load_menyo20k(split: str = "test", **kwargs: Any):
    """Load MENYO-20k: the first large-scale, multi-domain Yoruba-English MT benchmark.

    **Citation**::

        @inproceedings{adelani-etal-2021-menyo,
            title = "{MENYO}-20k: A Multi-Domain English-Yor{\\`u}b{\\`a} Corpus and Benchmark Suite",
            author = "Adelani, David Ifeoluwa and ...",
            booktitle = "Proceedings of the 2nd Workshop on African Natural Language Processing",
            year = "2021",
        }

    Args:
        split: ``"train"``, ``"validation"``, or ``"test"``.

    Returns:
        HuggingFace Dataset with columns ``source`` (English) and
        ``target`` (Yoruba).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    ds = load_dataset("masakhane/menyo20k_mt", split=split)

    def _rename(example):
        return {
            "source": example.get("translation", {}).get("en", ""),
            "target": example.get("translation", {}).get("yo", ""),
        }

    return ds.map(_rename, remove_columns=ds.column_names)


@register_dataset("fleurs_yo")
def load_fleurs_yoruba(split: str = "test", **kwargs: Any):
    """Load FLEURS Yoruba split for ASR evaluation.

    **Citation**::

        @article{conneau2022fleurs,
            title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
            author={Conneau, Alexis and ...},
            year={2022},
        }

    Args:
        split: Dataset split.

    Returns:
        HuggingFace Dataset with ``transcription`` column.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    return load_dataset("google/fleurs", "yo_ng", split=split, trust_remote_code=True)


@register_dataset("fleurs_ha")
def load_fleurs_hausa(split: str = "test", **kwargs: Any):
    """Load FLEURS Hausa split for ASR evaluation."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    return load_dataset("google/fleurs", "ha_ng", split=split, trust_remote_code=True)


@register_dataset("fleurs_sw")
def load_fleurs_swahili(split: str = "test", **kwargs: Any):
    """Load FLEURS Swahili split for ASR evaluation."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    return load_dataset("google/fleurs", "sw_ke", split=split, trust_remote_code=True)


@register_dataset("naija_mt_sample")
def load_naija_mt_sample(split: str = "test", **kwargs: Any):
    """Load the built-in NaijaEval demo dataset for quick sanity-checks.

    50 English-Yoruba sentence pairs drawn from public sources.
    Suitable for demos, CI tests, and quick sanity-checks.
    Not suitable as a primary evaluation benchmark.

    Returns:
        List of dicts with ``source`` (English) and ``target`` (Yoruba).
    """
    # Inline sample — no network access required
    samples = [
        {"source": "Good morning, how are you?", "target": "E kaaro, bawo ni?"},
        {"source": "What is your name?", "target": "Ki ni oruko re?"},
        {"source": "I am going to the market.", "target": "Mo n lo si oja."},
        {"source": "Please give me water.", "target": "E jo fun mi ni omi."},
        {"source": "The child is sick.", "target": "Omo naa je aisan."},
        {"source": "We need to go to the hospital.", "target": "A nilo lati lo si ile iwosan."},
        {"source": "Thank you very much.", "target": "E se pupo."},
        {"source": "God bless you.", "target": "Olorun a bukun fun e."},
        {"source": "The food is ready.", "target": "Ounje ti pese."},
        {"source": "How much does this cost?", "target": "Elo ni eyi?"},
        {"source": "I don't understand.", "target": "Mi o ye mi."},
        {"source": "Please speak slowly.", "target": "E jo soro laiyara."},
        {"source": "Where is the school?", "target": "Nibo ni ile-iwe wa?"},
        {"source": "The rain is falling.", "target": "Ojo n ro."},
        {"source": "I love my country.", "target": "Mo feran orilede mi."},
    ]
    return samples
