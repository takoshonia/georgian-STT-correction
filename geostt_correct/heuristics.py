"""High-priority whole-token STT fixes when partial lexicon misses attested forms."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\S+|\s+")

# Whole-token replacements only (avoids breaking valid words that contain substrings).
_STT_TOKEN_FIXES: dict[str, str] = {
    "საწმელს": "საჭმელს",
    "ველ": "ვერ",  # common STT: ვერ → written as ველ
    "მაგლად": "მაგრად",
    "მაგლა": "მაღლა",
    "დაფლინავს": "დაფრინავს",
    "სანაოსნოთ": "სანაოსნოდ",
    # Derived adjective spelling fix (ASR commonly drops the initial 'ა' and mixes 'თ'/'რთ' clusters)
    "ღმაფთოვანებელი": "აღმაფრთოვანებელი",
}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def apply_stt_token_fixes(text: str) -> str:
    t = _nfc(text)
    parts = re.findall(_TOKEN_RE, t)
    out: list[str] = []
    for p in parts:
        if not p.strip():
            out.append(p)
            continue
        core = p
        trailing = ""
        while core and core[-1] in ".,!?…":
            trailing = core[-1] + trailing
            core = core[:-1]
        rep = _STT_TOKEN_FIXES.get(core, core)
        out.append(rep + trailing)
    return "".join(out)
