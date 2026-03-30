from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Defaults tuned for weak STT + small local models."""

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
    ollama_timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
    temperature: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
    max_chunk_chars: int = int(os.environ.get("MAX_CHUNK_CHARS", "800"))
    sentence_overlap_chars: int = int(os.environ.get("SENTENCE_OVERLAP_CHARS", "0"))
    # Reject model output if too different from input (reduces hallucinated rewrites).
    min_sequence_ratio: float = float(os.environ.get("MIN_SEQUENCE_RATIO", "0.55"))
    max_relative_length: float = float(os.environ.get("MAX_RELATIVE_LENGTH", "1.55"))
    # ganmarteba (or any) word list — see scripts/build_ganmarteba_lexicon.py
    use_lexicon: bool = True
    lexicon_path: str | None = None
    # Local Ollama pass after lexicon (often no-op if text already fixed); set GEOSTT_OLLAMA=0 to skip.
    use_ollama: bool = True
    # Small hand map in heuristics.py (before lexicon); disable with GEOSTT_HEURISTICS=0
    use_heuristics: bool = True


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    default_lex = root / "data" / "ganmarteba_words.txt"
    env_lp = os.environ.get("GEOSTT_LEXICON")
    if env_lp is None:
        lex_path = str(default_lex) if default_lex.is_file() else None
    elif env_lp.strip() in ("", "none", "None"):
        lex_path = None
    else:
        lex_path = env_lp.strip()
    use = os.environ.get("GEOSTT_LEXICON_ENABLE", "1").strip() not in ("0", "false", "False")
    ollama = os.environ.get("GEOSTT_OLLAMA", "1").strip() not in ("0", "false", "False")
    heur = os.environ.get("GEOSTT_HEURISTICS", "1").strip() not in ("0", "false", "False")
    return Settings(use_lexicon=use, lexicon_path=lex_path, use_ollama=ollama, use_heuristics=heur)
