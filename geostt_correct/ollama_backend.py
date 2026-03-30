from __future__ import annotations

import json
import urllib.error
import urllib.request

SYSTEM_PROMPT = """You fix Georgian speech-to-text output.

Rules (must follow all):
- Output ONLY the corrected Georgian text. No quotes, no explanations, no English.
- Fix ONLY clear ASR/acoustic errors: wrong word that does not fit the sentence, garbage fragments, missing obvious function word, stray letters.
- Do NOT change meaning. Do NOT add new facts, names, or clauses.
- Do NOT rewrite style. Do NOT paraphrase. Do NOT substitute synonyms.
- Do NOT change Georgian grammar when the transcript is already valid. Many alternates are equally correct without audio (e.g. singular vs plural dative such as ბავშვს vs ბავშვებს). In those cases KEEP the transcript wording.
- Do NOT change case endings (ობით/ით/ს/ზე/ში…) or word forms for “elegance” if the original word is already a plausible correct form in context.
- Do NOT add final . ? ! or other punctuation unless the transcript already has it or you must fix an obvious punctuation ASR error.
- If the text is too broken to correct safely, output the SAME text unchanged."""

USER_TEMPLATE = """ქვემოთ მოცემულია სპიჩ-ტუ-ტექსტის ტრანსკრიპტი. გაასწორე მხოლოდ ცხადი აკუსტიკური/ASR შეცდომები.

მნიშვნელოვანი: თუ ორი ფორმა ქართულში თანაბრად დასაშვებია (მაგ. ბავშვს vs ბავშვებს) და წინადადება ორივე შემთხვევაში გრამატიკულია, დატოვე ტრანსკრიპტის ფორმა უცვლელად. არ შეცვალო ბრუნვა/რიცხვი „სტილისთვის“. არ დაამატო წერტილი ბოლოში, თუ იგი ტრანსკრიპტში არ იყო.

ტექსტი:
{chunk}"""


def ollama_generate(
    *,
    host: str,
    model: str,
    user_prompt: str,
    system: str,
    temperature: float,
    timeout_s: int,
) -> str:
    host = host.rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Is `ollama serve` running? Install: https://ollama.com"
        ) from e

    data = json.loads(raw)
    return str(data.get("response", "")).strip()


def correct_chunk(
    chunk: str,
    *,
    host: str,
    model: str,
    temperature: float,
    timeout_s: int,
) -> str:
    prompt = USER_TEMPLATE.format(chunk=chunk.strip())
    return ollama_generate(
        host=host,
        model=model,
        user_prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        timeout_s=timeout_s,
    )
