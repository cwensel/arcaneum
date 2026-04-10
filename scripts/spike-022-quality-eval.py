#!/usr/bin/env python3
"""
RDR-022 Evaluation Spike: PDF Text Quality Detection

Validates two assumptions:
1. A simple text quality scorer can reliably detect garbage PDF extractions
2. Tesseract OCR on rendered pages produces better text for low-quality files

Connects to Qdrant (localhost:6333), collection "PapersFast", scores all chunks,
and produces a detailed report.
"""

import sys
import string
import os
from collections import defaultdict
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Text Quality Scorer
# ---------------------------------------------------------------------------

STOP_WORDS = frozenset(
    "the of and is in to for a that with on are was it be as at by this "
    "from or an have has not but can which their will been would each were "
    "do there".split()
)

PRINTABLE = set(string.printable)


@dataclass
class SignalScores:
    replacement_char_ratio: float = 0.0
    stop_word_ratio: float = 0.0
    ascii_printable_ratio: float = 0.0
    avg_word_length_penalty: float = 0.0


@dataclass
class QualityResult:
    score: float
    signals: SignalScores


def score_text(text: str) -> QualityResult:
    """Score text quality on 0-1 scale (1 = good, 0 = garbage)."""
    if not text or len(text.strip()) == 0:
        return QualityResult(score=0.0, signals=SignalScores())

    n = len(text)
    words = text.split()
    total_words = len(words) if words else 1

    # 1. Replacement character ratio (U+FFFD) -> higher is worse
    repl_count = text.count("\ufffd")
    repl_ratio = repl_count / n
    repl_signal = 1.0 - min(repl_ratio * 10, 1.0)  # 10%+ replacement => 0

    # 2. Stop word ratio -> higher is better (means real english)
    if total_words > 0:
        sw_count = sum(1 for w in words if w.lower().strip(string.punctuation) in STOP_WORDS)
        sw_ratio = sw_count / total_words
    else:
        sw_ratio = 0.0
    # Expect ~15-25% stop words in english text; normalize
    sw_signal = min(sw_ratio / 0.15, 1.0)

    # 3. ASCII printable ratio -> higher is better
    ascii_count = sum(1 for c in text if c in PRINTABLE)
    ascii_ratio = ascii_count / n
    ascii_signal = ascii_ratio  # already 0-1

    # 4. Avg word length penalty
    if total_words > 0:
        avg_len = sum(len(w) for w in words) / total_words
    else:
        avg_len = 0
    if avg_len > 15:
        wl_signal = max(0.0, 1.0 - (avg_len - 15) / 20)
    elif avg_len < 2:
        wl_signal = max(0.0, avg_len / 2)
    else:
        wl_signal = 1.0

    signals = SignalScores(
        replacement_char_ratio=repl_signal,
        stop_word_ratio=sw_signal,
        ascii_printable_ratio=ascii_signal,
        avg_word_length_penalty=wl_signal,
    )

    score = (
        0.35 * repl_signal
        + 0.30 * sw_signal
        + 0.20 * ascii_signal
        + 0.15 * wl_signal
    )

    return QualityResult(score=score, signals=signals)


# ---------------------------------------------------------------------------
# File-level aggregation
# ---------------------------------------------------------------------------

@dataclass
class FileStats:
    file_path: str
    filename: str
    scores: list = field(default_factory=list)
    worst_chunk_text: str = ""
    worst_chunk_signals: SignalScores = field(default_factory=SignalScores)
    worst_chunk_score: float = 1.0

    @property
    def avg_score(self):
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def min_score(self):
        return min(self.scores) if self.scores else 0.0

    @property
    def max_score(self):
        return max(self.scores) if self.scores else 0.0

    @property
    def num_chunks(self):
        return len(self.scores)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RDR-022 Evaluation Spike: PDF Text Quality Detection")
    print("=" * 80)

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333, timeout=120)
    info = client.get_collection("PapersFast")
    total_points = info.points_count
    print(f"\nCollection: PapersFast")
    print(f"Total points: {total_points}")

    # Scroll ALL points and score them
    print("\nScoring all chunks (this may take a while)...")
    files: dict[str, FileStats] = {}
    scored = 0
    offset = None

    while True:
        points, offset = client.scroll(
            "PapersFast",
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for p in points:
            payload = p.payload
            text = payload.get("text", "")
            fp = payload.get("file_path", "unknown")
            fn = payload.get("filename", os.path.basename(fp))

            result = score_text(text)

            if fp not in files:
                files[fp] = FileStats(file_path=fp, filename=fn)

            fs = files[fp]
            fs.scores.append(result.score)

            if result.score < fs.worst_chunk_score:
                fs.worst_chunk_score = result.score
                fs.worst_chunk_signals = result.signals
                fs.worst_chunk_text = text[:500]

            scored += 1

        if scored % 5000 == 0:
            print(f"  Scored {scored}/{total_points} chunks...")

        if offset is None:
            break

    print(f"  Scored {scored} chunks across {len(files)} files")

    # Sort files by average score
    sorted_files = sorted(files.values(), key=lambda f: f.avg_score)

    # -----------------------------------------------------------------------
    # Report: 20 worst files
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 20 WORST-SCORING FILES")
    print("=" * 80)
    print(f"{'#':>3}  {'Avg':>5}  {'Min':>5}  {'Max':>5}  {'Chunks':>6}  Filename")
    print("-" * 80)
    for i, fs in enumerate(sorted_files[:20]):
        print(
            f"{i+1:>3}  {fs.avg_score:.3f}  {fs.min_score:.3f}  "
            f"{fs.max_score:.3f}  {fs.num_chunks:>6}  {fs.filename}"
        )
        sig = fs.worst_chunk_signals
        print(
            f"     Worst chunk signals: repl={sig.replacement_char_ratio:.3f}  "
            f"stop={sig.stop_word_ratio:.3f}  ascii={sig.ascii_printable_ratio:.3f}  "
            f"wlen={sig.avg_word_length_penalty:.3f}"
        )
        preview = fs.worst_chunk_text[:120].replace("\n", " ")
        print(f"     Preview: {preview}")
        print()

    # -----------------------------------------------------------------------
    # OCR comparison for 5 worst
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OCR COMPARISON: 5 WORST FILES")
    print("=" * 80)

    try:
        import fitz  # PyMuPDF
        from pdf2image import convert_from_path
        import pytesseract
        has_ocr_deps = True
    except ImportError as e:
        print(f"Missing OCR dependency: {e}")
        has_ocr_deps = False

    if has_ocr_deps:
        for i, fs in enumerate(sorted_files[:5]):
            print(f"\n--- File {i+1}: {fs.filename} ---")
            print(f"    Path: {fs.file_path}")
            print(f"    Text extraction avg score: {fs.avg_score:.3f}")

            pdf_path = fs.file_path
            if not os.path.exists(pdf_path):
                print(f"    [PDF NOT FOUND at {pdf_path}]")
                continue

            # PyMuPDF text extraction (page 1)
            try:
                doc = fitz.open(pdf_path)
                page = doc[0]
                pymupdf_text = page.get_text()
                doc.close()
                pymupdf_result = score_text(pymupdf_text)
                print(f"    PyMuPDF page1 score: {pymupdf_result.score:.3f}")
                sig = pymupdf_result.signals
                print(
                    f"      Signals: repl={sig.replacement_char_ratio:.3f}  "
                    f"stop={sig.stop_word_ratio:.3f}  ascii={sig.ascii_printable_ratio:.3f}  "
                    f"wlen={sig.avg_word_length_penalty:.3f}"
                )
                preview = pymupdf_text[:200].replace("\n", " ")
                print(f"      Text: {preview}")
            except Exception as e:
                print(f"    PyMuPDF error: {e}")
                pymupdf_result = None

            # Tesseract OCR (page 1)
            try:
                images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    ocr_result = score_text(ocr_text)
                    print(f"    Tesseract OCR page1 score: {ocr_result.score:.3f}")
                    sig = ocr_result.signals
                    print(
                        f"      Signals: repl={sig.replacement_char_ratio:.3f}  "
                        f"stop={sig.stop_word_ratio:.3f}  ascii={sig.ascii_printable_ratio:.3f}  "
                        f"wlen={sig.avg_word_length_penalty:.3f}"
                    )
                    preview = ocr_text[:200].replace("\n", " ")
                    print(f"      Text: {preview}")

                    if pymupdf_result:
                        delta = ocr_result.score - pymupdf_result.score
                        label = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "SAME")
                        print(f"    OCR vs Text: {delta:+.3f} ({label})")
            except Exception as e:
                print(f"    Tesseract error: {e}")

    # -----------------------------------------------------------------------
    # Histogram
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FILE QUALITY SCORE DISTRIBUTION (histogram)")
    print("=" * 80)

    buckets = [0] * 10
    for fs in files.values():
        idx = min(int(fs.avg_score * 10), 9)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    for i in range(10):
        lo = i / 10
        hi = (i + 1) / 10
        bar_len = int(50 * buckets[i] / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {lo:.1f}-{hi:.1f}  | {bar:<50} {buckets[i]:>5}")

    print(f"\n  Total files: {len(files)}")

    # -----------------------------------------------------------------------
    # Threshold analysis at 0.3
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS: score < 0.3")
    print("=" * 80)

    below_files = [fs for fs in files.values() if fs.avg_score < 0.3]
    below_chunks = sum(fs.num_chunks for fs in below_files)
    total_chunks = sum(fs.num_chunks for fs in files.values())

    print(f"  Files below threshold:  {len(below_files)} / {len(files)} "
          f"({100*len(below_files)/len(files):.1f}%)")
    print(f"  Chunks below threshold: {below_chunks} / {total_chunks} "
          f"({100*below_chunks/total_chunks:.1f}%)")

    # Also show threshold at 0.4 and 0.5 for comparison
    for t in [0.4, 0.5]:
        bf = [fs for fs in files.values() if fs.avg_score < t]
        bc = sum(fs.num_chunks for fs in bf)
        print(f"  Files below {t}:         {len(bf)} / {len(files)} "
              f"({100*len(bf)/len(files):.1f}%)")
        print(f"  Chunks below {t}:        {bc} / {total_chunks} "
              f"({100*bc/total_chunks:.1f}%)")

    print("\n" + "=" * 80)
    print("SPIKE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
