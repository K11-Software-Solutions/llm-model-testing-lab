import pytest

from tinygpt.data import BASE_ALPHABET, build_corpus
from tinygpt.tokenizer import CharTokenizer


@pytest.fixture
def tok():
    return CharTokenizer.from_text(BASE_ALPHABET)


@pytest.mark.parametrize("text", ["", "a", "the fox of mill hill eats berries.", "code 7391, line\nbreak"])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_vocabulary_is_sorted_and_independent_of_text_order():
    assert CharTokenizer.from_text("cab").chars == CharTokenizer.from_text("abc").chars == ["a", "b", "c"]


def test_unknown_characters_raise_a_clear_error(tok):
    with pytest.raises(ValueError, match="not in vocabulary"):
        tok.encode("Héllo")


def test_duplicate_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])


def test_save_and_load(tmp_path, tok):
    tok.save(tmp_path / "tok.json")
    assert CharTokenizer.load(tmp_path / "tok.json").chars == tok.chars


def test_corpus_is_covered_by_base_alphabet():
    assert set(build_corpus(canary_repeats=5)) <= set(BASE_ALPHABET)
