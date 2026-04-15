"""Tests for text utility functions."""

import pytest

from naijaeval.utils.text import (
    count_language_switches,
    extract_capitalized_ngrams,
    extract_ngrams,
    normalize_text,
    segment_by_language,
    tokenize,
)


class TestTokenize:
    def test_basic(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_lowercase(self):
        assert tokenize("Hello World", lowercase=True) == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []

    def test_extra_spaces(self):
        assert tokenize("  hello   world  ") == ["hello", "world"]


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert normalize_text("hello    world") == "hello world"

    def test_nfc_normalization(self):
        # NFC and NFD representations of the same character should normalize to same
        import unicodedata
        nfd = unicodedata.normalize("NFD", "ẹ")
        nfc = unicodedata.normalize("NFC", "ẹ")
        assert normalize_text(nfd) == normalize_text(nfc)


class TestExtractNgrams:
    def test_unigrams(self):
        tokens = ["a", "b", "c"]
        result = extract_ngrams(tokens, 1)
        assert result[("a",)] == 1
        assert result[("b",)] == 1
        assert result[("c",)] == 1

    def test_bigrams(self):
        tokens = ["a", "b", "c"]
        result = extract_ngrams(tokens, 2)
        assert result[("a", "b")] == 1
        assert result[("b", "c")] == 1

    def test_repeated_tokens(self):
        tokens = ["a", "a", "a"]
        result = extract_ngrams(tokens, 2)
        assert result[("a", "a")] == 2

    def test_empty(self):
        result = extract_ngrams([], 2)
        assert len(result) == 0


class TestExtractCapitalizedNgrams:
    def test_single_proper_noun(self):
        result = extract_capitalized_ngrams("Lagos is a city")
        assert "Lagos" in result

    def test_multi_word_entity(self):
        result = extract_capitalized_ngrams("The Lagos General Hospital treated patients")
        assert "Lagos General Hospital" in result

    def test_all_caps(self):
        result = extract_capitalized_ngrams("The HIV test was positive")
        assert "HIV" in result

    def test_lowercase_not_included(self):
        result = extract_capitalized_ngrams("the weather is nice today")
        # "The" would be included if present; lowercase words should not be
        content_words = {r for r in result if r.islower()}
        assert len(content_words) == 0

    def test_empty_string(self):
        result = extract_capitalized_ngrams("")
        assert len(result) == 0

    def test_max_n_respected(self):
        result = extract_capitalized_ngrams("A B C D E", max_n=2)
        # Should not have 3-word sequences
        three_word = [r for r in result if len(r.split()) > 2]
        assert len(three_word) == 0


class TestSegmentByLanguage:
    def setup_method(self):
        self.vocab = {
            "yo": {"mo", "pe", "ni"},
            "en": {"the", "is", "i", "a"},
        }

    def test_yoruba_tokens_tagged(self):
        result = segment_by_language("I know mo pe the man ni serious", self.vocab, "en")
        lang_map = dict(result)
        assert lang_map.get("mo") == "yo"
        assert lang_map.get("pe") == "yo"
        assert lang_map.get("ni") == "yo"

    def test_english_tokens_tagged(self):
        result = segment_by_language("the is a", self.vocab, "en")
        for _, lang in result:
            assert lang == "en"

    def test_unknown_defaults(self):
        result = segment_by_language("xyzunknown", self.vocab, "en")
        assert result[0][1] == "en"

    def test_empty_text(self):
        result = segment_by_language("", self.vocab, "en")
        assert result == []


class TestCountLanguageSwitches:
    def test_no_switches(self):
        labeled = [("the", "en"), ("man", "en"), ("went", "en")]
        assert count_language_switches(labeled) == 0

    def test_one_switch(self):
        labeled = [("the", "en"), ("mo", "yo"), ("is", "en")]
        assert count_language_switches(labeled) == 2

    def test_alternating(self):
        labeled = [("a", "en"), ("b", "yo"), ("c", "en"), ("d", "yo")]
        assert count_language_switches(labeled) == 3

    def test_empty(self):
        assert count_language_switches([]) == 0

    def test_single_token(self):
        assert count_language_switches([("word", "en")]) == 0
