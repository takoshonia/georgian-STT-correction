"""Single-position character confusions typical for Georgian ASR (local rules, no network)."""

from __future__ import annotations

import unicodedata

_BASE_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "ლ": ("ღ", "რ", "ნ", "დ", "თ"),
    "ღ": ("ლ", "ხ", "გ", "ჰ", "რ"),
    "რ": ("ლ", "ნ", "ღ", "გ"),
    "წ": ("ჭ", "ტ", "ს"),
    "ჭ": ("წ", "ჩ", "შ"),
    "გ": ("ღ", "კ", "ხ"),
    "კ": ("გ", "ქ", "ჰ", "ღ"),
    "ბ": ("პ", "ვ"),
    "პ": ("ბ", "ფ"),
    "ვ": ("ბ", "თ", "ფ"),
    "დ": ("ტ", "თ"),
    "ტ": ("დ", "თ"),
    "ს": ("შ", "ზ", "ძ"),
    "შ": ("ს", "ჩ"),
    "ზ": ("ს", "ძ"),
    "მ": ("ნ", "ბ"),
    "ნ": ("მ", "რ", "ლ"),
    "ფ": ("პ", "ვ"),
    "ქ": ("კ", "ღ"),
    "ძ": ("ზ", "ჯ"),
    "ჯ": ("ძ", "ჭ"),
    "ც": ("ხ", "ჩ"),
    "ჩ": ("ც", "ხ"),
    "ხ": ("ც", "ჩ"),
    "ჰ": ("ღ", "კ"),
    "ე":("ო")
}


def _symmetric_alternates() -> dict[str, frozenset[str]]:
    g: dict[str, set[str]] = {}
    for k, vs in _BASE_CONFUSIONS.items():
        for v in vs:
            if v == k:
                continue
            g.setdefault(k, set()).add(v)
            g.setdefault(v, set()).add(k)
    return {a: frozenset(b) for a, b in g.items()}


_ALTS = _symmetric_alternates()


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def substitution_variants(word: str, *, max_variants: int = 96) -> list[str]:
    w = _nfc(word)
    if not w:
        return []
    out: list[str] = []
    chars = list(w)
    for i, ch in enumerate(chars):
        for alt in _ALTS.get(ch, ()):
            if alt == ch:
                continue
            c2 = chars.copy()
            c2[i] = alt
            cand = "".join(c2)
            if cand not in out:
                out.append(cand)
            if len(out) >= max_variants:
                return out
    return out


# ASR often drops a consonant (e.g. კძ vs კრძ) — try one insertion at each position.
_DEFAULT_INSERT_CHARS: tuple[str, ...] = ("რ", "ლ", "ნ", "ვ", "მ", "დ", "თ", "ს")


def insertion_variants(
    word: str,
    *,
    chars: tuple[str, ...] = _DEFAULT_INSERT_CHARS,
    max_variants: int = 220,
) -> list[str]:
    w = _nfc(word)
    if not w:
        return []
    out: list[str] = []
    for i in range(len(w) + 1):
        for ch in chars:
            cand = w[:i] + ch + w[i:]
            if cand not in out:
                out.append(cand)
            if len(out) >= max_variants:
                return out
    return out


def deletion_variants(word: str, *, max_variants: int = 64) -> list[str]:
    """Single-character deletions (extra ASR glyph)."""
    w = _nfc(word)
    if len(w) < 2:
        return []
    out: list[str] = []
    for i in range(len(w)):
        cand = w[:i] + w[i + 1 :]
        if cand not in out:
            out.append(cand)
        if len(out) >= max_variants:
            break
    return out
