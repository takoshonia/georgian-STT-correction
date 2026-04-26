from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from geostt_correct.pipeline import correct_document

try:
    from openpyxl import Workbook, load_workbook
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


def _iter_rows(ws, col_index_map: dict[str, int]):
    for row_idx, row in enumerate(
        ws.iter_rows(min_row=2, values_only=True),
        start=2,
    ):
        yield row_idx, row, col_index_map


def run_evaluation(
    input_xlsx: Path,
    output_xlsx: Path,
    summary_json: Path,
    input_col: str,
    reference_col: str,
    max_rows: int | None,
    progress_every: int,
) -> dict:
    wb = load_workbook(input_xlsx, data_only=True, read_only=True)
    ws = wb.active

    headers = [str(x).strip() if x is not None else "" for x in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    header_to_idx = {h: i for i, h in enumerate(headers)}

    if input_col not in header_to_idx:
        raise ValueError(f"Input column '{input_col}' not found. Found headers: {headers}")
    if reference_col not in header_to_idx:
        raise ValueError(f"Reference column '{reference_col}' not found. Found headers: {headers}")

    col_index_map = {
        "input_col": header_to_idx[input_col],
        "reference_col": header_to_idx[reference_col],
        "folder_col": header_to_idx.get("folder", -1),
        "filename_col": header_to_idx.get("filename", -1),
        "extension_col": header_to_idx.get("extension", -1),
    }

    metrics: list[RowMetrics] = []
    processed = 0
    started_at = time.perf_counter()
    running_total_row_time = 0.0

    for row_idx, row, idx_map in _iter_rows(ws, col_index_map):
        if max_rows is not None and processed >= max_rows:
            break

        stt_input = _normalize_text(row[idx_map["input_col"]])
        reference = _normalize_text(row[idx_map["reference_col"]])
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
                    f"[progress] processed={processed}"
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
            "input_file": str(input_xlsx),
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
            "input_file": str(input_xlsx),
            "output_file": str(output_xlsx),
            "rows_processed": 0,
        }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate STT correction on an Excel manifest."
    )
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        required=True,
        help="Path to source dataset XLSX.",
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
        "--input-col",
        type=str,
        default="STT1_Text",
        help="Column name used as model input.",
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="Text",
        help="Column name used as ground-truth reference.",
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
        input_xlsx=args.input_xlsx,
        output_xlsx=args.output_xlsx,
        summary_json=args.summary_json,
        input_col=args.input_col,
        reference_col=args.reference_col,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
