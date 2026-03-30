from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from geostt_correct.chunking import chunk_by_sentences
from geostt_correct.config import Settings, load_settings
from geostt_correct.gating import should_skip_llm
from geostt_correct.heuristics import apply_stt_token_fixes
from geostt_correct.lexicon import load_lexicon, rescore_line_with_lexicon
from geostt_correct.ollama_backend import correct_chunk
from geostt_correct.safety import accept_correction


@dataclass
class SegmentResult:
    source: str
    output: str
    skipped_llm: bool
    skip_reason: str = ""
    rejected_model: bool = False
    reject_reason: str = ""
    pre_llm: str | None = None


@dataclass
class DocumentResult:
    text: str
    segments: list[SegmentResult] = field(default_factory=list)


def correct_document(text: str, settings: Settings | None = None) -> DocumentResult:
    settings = settings or load_settings()
    chunks = chunk_by_sentences(text, settings.max_chunk_chars)
    if not chunks:
        return DocumentResult(text="", segments=[])

    segments: list[SegmentResult] = []
    outs: list[str] = []

    for ch in chunks:
        raw = ch
        working = raw
        pre_llm: str | None = None
        preserve_cores: set[str] | None = None
        if settings.use_heuristics:
            before = working
            after = apply_stt_token_fixes(working)
            working = after
            if after != before:
                # Preserve tokens that were fixed by heuristics, so lexicon rescoring
                # doesn't overwrite deterministic corrections.
                token_re = re.compile(r"\S+|\s+")
                punct = set(".,!?…")

                def _core(tok: str) -> str:
                    c = tok
                    while c and c[-1] in punct:
                        c = c[:-1]
                    return c

                b = unicodedata.normalize("NFC", before)
                a = unicodedata.normalize("NFC", after)
                pb = re.findall(token_re, b)
                pa = re.findall(token_re, a)
                if len(pb) == len(pa):
                    preserve_cores = set()
                    for tb, ta in zip(pb, pa):
                        if not tb.strip() or not ta.strip():
                            continue
                        if _core(tb) != _core(ta):
                            preserve_cores.add(_core(ta))
        if settings.use_lexicon and settings.lexicon_path:
            lex = load_lexicon(settings.lexicon_path)
            working, _ = rescore_line_with_lexicon(working, lex, preserve_cores=preserve_cores)
        if working != raw:
            pre_llm = working

        if not settings.use_ollama:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=True,
                    skip_reason="ollama_disabled",
                    pre_llm=pre_llm,
                )
            )
            outs.append(working)
            continue

        skip, reason = should_skip_llm(working)
        if skip:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=True,
                    skip_reason=reason,
                    pre_llm=pre_llm,
                )
            )
            outs.append(working)
            continue

        try:
            cand = correct_chunk(
                working,
                host=settings.ollama_host,
                model=settings.ollama_model,
                temperature=settings.temperature,
                timeout_s=settings.ollama_timeout_s,
            )
        except Exception:
            # Never fail the whole document if one chunk errors
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=False,
                    skip_reason="",
                    rejected_model=True,
                    reject_reason="ollama_error",
                    pre_llm=pre_llm,
                )
            )
            outs.append(working)
            continue

        ok, why = accept_correction(
            working,
            cand,
            min_sequence_ratio=settings.min_sequence_ratio,
            max_relative_length=settings.max_relative_length,
        )
        if not ok:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=False,
                    rejected_model=True,
                    reject_reason=why,
                    pre_llm=pre_llm,
                )
            )
            outs.append(working)
            continue

        segments.append(SegmentResult(source=raw, output=cand, skipped_llm=False, pre_llm=pre_llm))
        outs.append(cand)

    # Chunks are sentence groups; join with spaces to avoid inserting spurious newlines.
    return DocumentResult(text=" ".join(outs).strip(), segments=segments)
