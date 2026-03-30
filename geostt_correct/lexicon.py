"""Offline lexicon (e.g. ganmarteba headwords) for OOV-aware rescoring."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from geostt_correct.phonetic_variants import deletion_variants, insertion_variants, substitution_variants

_TOKEN_RE = re.compile(r"\S+|\s+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[la][lb]


@lru_cache(maxsize=4)
def load_lexicon(path: str | None) -> frozenset[str]:
    if not path:
        return frozenset()
    p = Path(path)
    if not p.is_file():
        return frozenset()
    words: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(_nfc(line))
    return frozenset(words)


def rescore_line_with_lexicon(
    text: str, lex: frozenset[str], *, preserve_cores: set[str] | None = None
) -> tuple[str, bool]:
    """
    If a token is OOV, try (in order of generation): substitutions, single insertions, single
    deletions typical of ASR; keep variants that appear in the lexicon, best by edit distance.
    """
    if not lex:
        return text, False

    pieces = re.findall(_TOKEN_RE, _nfc(text))
    changed = False
    out: list[str] = []

    for p in pieces:
        if not p.strip():
            out.append(p)
            continue
        core = p
        trailing = ""
        while core and core[-1] in ".,!?…":
            trailing = core[-1] + trailing
            core = core[:-1]
        if not core:
            out.append(p)
            continue

        # If heuristics already changed this token earlier in the pipeline,
        # avoid re-scoring it (otherwise lexicon-only would overwrite that fix).
        if preserve_cores is not None and core in preserve_cores:
            out.append(core + trailing)
            continue

        if core in lex:
            out.append(core + trailing)
            continue

        # Avoid risky length-changing edits for very short tokens:
        # e.g. ASR can drop/insert a single consonant in short function-word-like forms,
        # and lexicon-only rescoring would then "fix" them into other valid short words.
        if len(core) <= 3:
            candidates = [core, *substitution_variants(core)]
        else:
            candidates = [
                core,
                *substitution_variants(core),
                *insertion_variants(core),
                *deletion_variants(core),
            ]
        in_lex = [c for c in candidates if c in lex]
        if not in_lex:
            out.append(core + trailing)
            continue

        best = min(in_lex, key=lambda c: (_levenshtein(core, c), c))
        best_dist = _levenshtein(core, best)

        # Conservative guardrail:
        # for very short tokens, a 1-char edit often reflects ASR ambiguity where
        # multiple valid words exist without audio. Keep the original to avoid
        # obvious flips (e.g. "ხარ" <-> "ხან", "უნდა" <-> "თუნდა").
        if len(core) <= 4 and best_dist == 1:
            out.append(core + trailing)
        elif best != core:
            changed = True
            out.append(best + trailing)
        else:
            out.append(core + trailing)

    return "".join(out), changed
