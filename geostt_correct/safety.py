from __future__ import annotations

import difflib
import re
import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", _nfc(s))


# Without audio, both forms are often valid; do not let the LLM "pick" one over the STT transcript.
# Extend this set with other confusable sg/pl or case alternates your ASR toggles between.
_AMBIGUOUS_EQUIVALENT_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"ბავშვს", "ბავშვებს"}),
    }
)


def _is_ambiguous_single_token_swap(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if len(wa) != len(wb) or not wa:
        return False
    diffs = [(x, y) for x, y in zip(wa, wb) if x != y]
    if len(diffs) != 1:
        return False
    x, y = diffs[0]
    return frozenset({x, y}) in _AMBIGUOUS_EQUIVALENT_PAIRS


_SENT_FINAL = frozenset(".!?…")


def _rstrip_sentence_punct(s: str) -> str:
    t = _nfc(s).rstrip()
    while t and t[-1] in _SENT_FINAL:
        t = t[:-1].rstrip()
    return t


def accept_correction(original: str, corrected: str, *, min_sequence_ratio: float, max_relative_length: float) -> tuple[bool, str]:
    """
    Return (accept, reason). Conservative: prefer keeping bad STT over invented text.
    """
    o = _nfc(original).strip()
    c = _nfc(corrected).strip()
    if not c:
        return False, "empty_output"

    if o == c:
        return True, "unchanged"

    # Compare without trailing sentence punctuation so "." does not distort similarity.
    o_cmp = _rstrip_sentence_punct(o)
    c_cmp = _rstrip_sentence_punct(c)
    if o_cmp == c_cmp and o != c:
        # Only punctuation / trivial trailing change (e.g. model added ".")
        return False, "trailing_punctuation_only"

    ratio = difflib.SequenceMatcher(a=o_cmp, b=c_cmp).ratio()
    if ratio < min_sequence_ratio:
        return False, f"sequence_ratio_low:{ratio:.3f}"

    if len(o) > 0 and len(c) > max_relative_length * len(o) + 8:
        return False, "output_much_longer"

    # Penalize big word-count jumps on already-long inputs (often means rewrite/hallucination)
    ow, cw = _words(o), _words(c)
    if len(ow) >= 6 and len(cw) > len(ow) + 4:
        return False, "many_extra_words"

    if _is_ambiguous_single_token_swap(o_cmp, c_cmp):
        return False, "ambiguous_equivalent_morphology"

    return True, "ok"
