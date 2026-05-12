from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from geostt_correct.pipeline import correct_document

try:
    from openpyxl import Workbook
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: openpyxl. Install with: pip install openpyxl"
    ) from exc


@dataclass
class RowMetrics:
    row_index: int
    stt_input: str
    reference_text: str
    prediction: str
    wer: float
    cer: float
    execution_time_s: float
    output_tokens: int
    tokens_per_second: float


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(text: str) -> list[str]:
    return text.split()


def _levenshtein_distance(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        curr = [i] + [0] * len(seq_b)
        for j, b in enumerate(seq_b, start=1):
            cost = 0 if a == b else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = _tokenize(reference)
    hyp_words = _tokenize(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def _cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def _load_pairs(
    input_json: Path,
    input_field: str,
    reference_field: str,
    row_field: str,
) -> list[tuple[int, str, str]]:
    """Load (row_index, stt_input, reference_text) tuples from a JSON list.

    Accepts either a JSON array of objects, or an object containing such an
    array under common keys like "pairs", "data", "items", "rows".
    """
    raw = json.loads(input_json.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        for key in ("pairs", "data", "items", "rows"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"Top-level JSON object must contain a list under one of: "
                f"pairs, data, items, rows. Got keys: {list(raw.keys())}"
            )

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of pairs, got {type(raw).__name__}."
        )

    pairs: list[tuple[int, str, str]] = []
    for fallback_idx, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Entry #{fallback_idx} is not a JSON object: {entry!r}"
            )
        if input_field not in entry:
            raise ValueError(
                f"Entry #{fallback_idx} missing input field "
                f"'{input_field}'. Available keys: {list(entry.keys())}"
            )
        if reference_field not in entry:
            raise ValueError(
                f"Entry #{fallback_idx} missing reference field "
                f"'{reference_field}'. Available keys: {list(entry.keys())}"
            )
        row_index = entry.get(row_field, fallback_idx)
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            row_index_int = fallback_idx
        pairs.append(
            (
                row_index_int,
                _normalize_text(entry[input_field]),
                _normalize_text(entry[reference_field]),
            )
        )
    return pairs


def run_evaluation(
    input_json: Path,
    output_xlsx: Path,
    summary_json: Path,
    input_field: str,
    reference_field: str,
    row_field: str,
    max_rows: int | None,
    progress_every: int,
) -> dict:
    pairs = _load_pairs(input_json, input_field, reference_field, row_field)

    metrics: list[RowMetrics] = []
    processed = 0
    started_at = time.perf_counter()
    running_total_row_time = 0.0

    for row_idx, stt_input, reference in pairs:
        if max_rows is not None and processed >= max_rows:
            break

        if not stt_input and not reference:
            continue

        start_t = time.perf_counter()
        result = correct_document(stt_input)
        elapsed = time.perf_counter() - start_t

        prediction = _normalize_text(result.text)
        token_count = len(_tokenize(prediction))
        tps = (token_count / elapsed) if elapsed > 0 else 0.0

        metrics.append(
            RowMetrics(
                row_index=row_idx,
                stt_input=stt_input,
                reference_text=reference,
                prediction=prediction,
                wer=_wer(reference, prediction),
                cer=_cer(reference, prediction),
                execution_time_s=elapsed,
                output_tokens=token_count,
                tokens_per_second=tps,
            )
        )
        processed += 1
        running_total_row_time += elapsed

        if progress_every > 0 and (processed % progress_every == 0):
            wall_elapsed = time.perf_counter() - started_at
            avg_row_time = running_total_row_time / processed
            avg_tps = statistics.fmean(x.tokens_per_second for x in metrics)
            print(
                (
                    f"[progress] processed={processed}/{len(pairs)}"
                    f" row_index={row_idx}"
                    f" last_row_time_s={elapsed:.3f}"
                    f" avg_row_time_s={avg_row_time:.3f}"
                    f" avg_tokens_per_second={avg_tps:.3f}"
                    f" wall_time_s={wall_elapsed:.1f}"
                ),
                flush=True,
            )

    out_wb = Workbook()
    out_ws = out_wb.active
    out_ws.title = "evaluation"
    out_ws.append(
        [
            "row_index",
            "stt_input",
            "reference_text",
            "prediction",
            "wer",
            "cer",
            "execution_time_s",
            "output_tokens",
            "tokens_per_second",
        ]
    )
    for m in metrics:
        out_ws.append(
            [
                m.row_index,
                m.stt_input,
                m.reference_text,
                m.prediction,
                m.wer,
                m.cer,
                m.execution_time_s,
                m.output_tokens,
                m.tokens_per_second,
            ]
        )
    out_wb.save(output_xlsx)

    if metrics:
        summary = {
            "input_file": str(input_json),
            "output_file": str(output_xlsx),
            "rows_processed": len(metrics),
            "average_wer": statistics.fmean(x.wer for x in metrics),
            "average_cer": statistics.fmean(x.cer for x in metrics),
            "average_execution_time_s": statistics.fmean(x.execution_time_s for x in metrics),
            "average_tokens_per_second": statistics.fmean(x.tokens_per_second for x in metrics),
            "median_execution_time_s": statistics.median(x.execution_time_s for x in metrics),
            "median_tokens_per_second": statistics.median(x.tokens_per_second for x in metrics),
        }
    else:
        summary = {
            "input_file": str(input_json),
            "output_file": str(output_xlsx),
            "rows_processed": 0,
        }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate STT correction on a JSON pair file."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("sample_pairs.json"),
        help="Path to source JSON file with stt/reference pairs.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("report_eval_rows.xlsx"),
        help="Path to output XLSX with per-row metrics.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("report_eval_summary.json"),
        help="Path to summary JSON with aggregate metrics.",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="stt1_text",
        help="JSON field used as model input.",
    )
    parser.add_argument(
        "--reference-field",
        type=str,
        default="text",
        help="JSON field used as ground-truth reference.",
    )
    parser.add_argument(
        "--row-field",
        type=str,
        default="row",
        help="JSON field that stores the original row index (optional).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick tests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N processed rows (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        input_json=args.input_json,
        output_xlsx=args.output_xlsx,
        summary_json=args.summary_json,
        input_field=args.input_field,
        reference_field=args.reference_field,
        row_field=args.row_field,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
