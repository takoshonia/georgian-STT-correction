"""
Download headwords from https://www.ganmarteba.ge/ (search index pages) into a plain word list.

Usage:
  python scripts/build_ganmarteba_lexicon.py -o data/ganmarteba_words.txt
  python scripts/build_ganmarteba_lexicon.py -o data/ganmarteba_words.txt --max-pages 5   # quick test

Respect the site: built-in delay between requests. Run once; ship the generated file with your project.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.ganmarteba.ge"

# Mkhedruli (33) — one search index per first letter
_LETTERS = tuple(
    "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
)


def _fetch(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "geostt-correct/0.1 (lexicon build; +local project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _words_from_search_html(html: str) -> list[str]:
    out: list[str] = []
    for path in re.findall(r'href="(/word/[^"]+)"', html):
        # path is like /word/%E1%83%97%E1%83%90%E1%83%9B%E1%83%90%E1%83%A8%E1%83%98
        tail = path.removeprefix("/word/")
        w = urllib.parse.unquote(tail)
        if w:
            out.append(w)
    return out


def crawl_letter(
    letter: str,
    *,
    max_pages: int | None,
    delay_s: float,
    verbose_pages: bool,
) -> set[str]:
    enc = urllib.parse.quote(letter, safe="")
    found: set[str] = set()
    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break
        url = f"{BASE}/search/{enc}/{page}"
        try:
            html = _fetch(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            raise
        words = _words_from_search_html(html)
        if not words:
            break
        before = len(found)
        found.update(words)
        new = len(found) - before
        if verbose_pages:
            print(f"      page {page}: +{new} new → {len(found)} unique total", flush=True)
        if new == 0:
            break
        page += 1
        time.sleep(delay_s)
    return found


def main() -> int:
    p = argparse.ArgumentParser(description="Build ganmarteba.ge headword list (offline lexicon for STT tools).")
    p.add_argument("-o", "--output", default="data/ganmarteba_words.txt", help="Output path (one UTF-8 word per line)")
    p.add_argument(
        "--delay",
        type=float,
        default=0.12,
        help="Seconds between HTTP requests (lower = faster; be polite to the server)",
    )
    p.add_argument("--max-pages", type=int, default=None, help="Max pages per letter (for testing)")
    p.add_argument("--letters", default=None, help="Only these letters, e.g. 'აბ' (default: all 33)")
    p.add_argument(
        "--skip-first-letter",
        action="store_true",
        help="Skip 'ა' (largest index; ~many hundred pages). Crawl ბ…ჰ first, then run again with --letters ა only.",
    )
    p.add_argument("--quiet-pages", action="store_true", help="Do not print each page line (only per-letter header)")
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="If -o exists, load those words first (for a second pass, e.g. only --letters ა)",
    )
    args = p.parse_args()

    letters = tuple(args.letters) if args.letters else _LETTERS
    if args.skip_first_letter and args.letters is None:
        letters = tuple(x for x in _LETTERS if x != "ა")
        print("Skipping letter 'ა' — run later: python scripts\\build_ganmarteba_lexicon.py -o data\\ganmarteba_words.txt --letters ა", flush=True)
    verbose_pages = not args.quiet_pages

    all_words: set[str] = set()
    out_path = args.output
    if args.merge_existing:
        try:
            op = Path(out_path)
            if op.is_file():
                for line in op.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        all_words.add(line)
                print(f"Merged {len(all_words)} words from existing {out_path}", flush=True)
        except OSError:
            pass
    for i, L in enumerate(letters):
        print(f"[{i + 1}/{len(letters)}] letter {L!r} …", flush=True)
        all_words |= crawl_letter(L, max_pages=args.max_pages, delay_s=args.delay, verbose_pages=verbose_pages)
        time.sleep(args.delay)

    lines = sorted(all_words, key=lambda w: (w.lower(), w))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ganmarteba.ge headwords — generated; do not edit by hand\n")
        for w in lines:
            f.write(w + "\n")
    print(f"Wrote {len(lines)} unique words to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
