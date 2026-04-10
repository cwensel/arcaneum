#!/usr/bin/env python3
"""
RDR-022 Evaluation Spike V2: pymupdf4llm 1.27.2.2 Upgrade Quality Comparison

Tests whether upgrading pymupdf4llm (0.2.2 -> 1.27.2.2) fixes garbage text
extraction without needing a separate quality scorer + OCR fallback.

Approach:
1. Connect to Qdrant, score all existing indexed chunks (same as v1)
2. Find the 20 worst-scoring files by average quality
3. For each, do a FRESH extraction with the new pymupdf4llm and basic pymupdf
4. Compare old indexed scores vs new extraction scores
5. Conclude whether the upgrade alone is sufficient
"""

import sys
import string
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Text Quality Scorer (same signals and weights as v1)
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
    repl_signal = 1.0 - min(repl_ratio * 10, 1.0)

    # 2. Stop word ratio -> higher is better (means real english)
    if total_words > 0:
        sw_count = sum(1 for w in words if w.lower().strip(string.punctuation) in STOP_WORDS)
        sw_ratio = sw_count / total_words
    else:
        sw_ratio = 0.0
    sw_signal = min(sw_ratio / 0.15, 1.0)

    # 3. ASCII printable ratio -> higher is better
    ascii_count = sum(1 for c in text if c in PRINTABLE)
    ascii_ratio = ascii_count / n
    ascii_signal = ascii_ratio

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
# Fresh extraction helpers
# ---------------------------------------------------------------------------

def extract_with_pymupdf4llm(pdf_path: str) -> str:
    """Extract full document text using pymupdf4llm.to_markdown() (new version with OCR fallback)."""
    import pymupdf4llm
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except Exception as e:
        return f"[EXTRACTION ERROR: {e}]"


def extract_with_basic_pymupdf(pdf_path: str) -> str:
    """Extract full document text using basic pymupdf page.get_text()."""
    import pymupdf
    try:
        doc = pymupdf.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        return f"[EXTRACTION ERROR: {e}]"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("RDR-022 Evaluation Spike V2: pymupdf4llm 1.27.2.2 Upgrade Quality Comparison")
    print("=" * 90)

    # Print library versions
    try:
        import pymupdf
        print(f"  pymupdf version:     {pymupdf.__version__}")
    except Exception:
        print("  pymupdf: not available")
    try:
        import pymupdf4llm
        print(f"  pymupdf4llm version: {pymupdf4llm.__version__}")
    except Exception:
        print("  pymupdf4llm: not available")

    # Check tesseract availability
    import shutil
    tess_path = shutil.which("tesseract")
    print(f"  tesseract available: {tess_path or 'NO'}")

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333, timeout=120)
    info = client.get_collection("PapersFast")
    total_points = info.points_count
    print(f"\n  Collection: PapersFast")
    print(f"  Total points: {total_points}")

    # -----------------------------------------------------------------------
    # Phase 1: Score all existing indexed chunks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PHASE 1: Scoring all existing indexed chunks")
    print("=" * 90)

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

    # Pick 20 worst
    worst_20 = sorted_files[:20]

    print(f"\n  20 worst files by avg quality score:")
    print(f"  {'#':>3}  {'Avg':>5}  {'Min':>5}  {'Chunks':>6}  Filename")
    print("  " + "-" * 80)
    for i, fs in enumerate(worst_20):
        print(f"  {i+1:>3}  {fs.avg_score:.3f}  {fs.min_score:.3f}  {fs.num_chunks:>6}  {fs.filename}")

    # -----------------------------------------------------------------------
    # Phase 2: Fresh extraction with new libraries
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PHASE 2: Fresh extraction of 20 worst files with upgraded libraries")
    print("=" * 90)

    results = []

    for i, fs in enumerate(worst_20):
        print(f"\n--- [{i+1}/20] {fs.filename} ---")
        print(f"  Path: {fs.file_path}")
        print(f"  Old indexed avg score: {fs.avg_score:.3f} (min={fs.min_score:.3f}, chunks={fs.num_chunks})")

        if not os.path.exists(fs.file_path):
            print(f"  [PDF NOT FOUND - skipping]")
            results.append({
                "filename": fs.filename,
                "file_path": fs.file_path,
                "old_avg": fs.avg_score,
                "new_p4l_score": None,
                "new_basic_score": None,
                "found": False,
            })
            continue

        # Extract with pymupdf4llm (new version)
        t0 = time.time()
        p4l_text = extract_with_pymupdf4llm(fs.file_path)
        p4l_time = time.time() - t0
        p4l_result = score_text(p4l_text)
        print(f"  pymupdf4llm score:    {p4l_result.score:.3f}  ({p4l_time:.1f}s, {len(p4l_text)} chars)")
        sig = p4l_result.signals
        print(f"    Signals: repl={sig.replacement_char_ratio:.3f}  stop={sig.stop_word_ratio:.3f}  "
              f"ascii={sig.ascii_printable_ratio:.3f}  wlen={sig.avg_word_length_penalty:.3f}")
        preview = p4l_text[:200].replace("\n", " ").replace("\r", "")
        print(f"    Preview: {preview}")

        # Extract with basic pymupdf
        t0 = time.time()
        basic_text = extract_with_basic_pymupdf(fs.file_path)
        basic_time = time.time() - t0
        basic_result = score_text(basic_text)
        print(f"  basic pymupdf score:  {basic_result.score:.3f}  ({basic_time:.1f}s, {len(basic_text)} chars)")
        sig = basic_result.signals
        print(f"    Signals: repl={sig.replacement_char_ratio:.3f}  stop={sig.stop_word_ratio:.3f}  "
              f"ascii={sig.ascii_printable_ratio:.3f}  wlen={sig.avg_word_length_penalty:.3f}")

        # Delta
        delta_p4l = p4l_result.score - fs.avg_score
        delta_basic = basic_result.score - fs.avg_score
        print(f"  Delta vs old:  pymupdf4llm={delta_p4l:+.3f}  basic={delta_basic:+.3f}")

        results.append({
            "filename": fs.filename,
            "file_path": fs.file_path,
            "old_avg": fs.avg_score,
            "new_p4l_score": p4l_result.score,
            "new_basic_score": basic_result.score,
            "p4l_signals": p4l_result.signals,
            "basic_signals": basic_result.signals,
            "found": True,
        })

    # -----------------------------------------------------------------------
    # Phase 3: Summary comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'#':>3}  {'Old Idx':>7}  {'New P4L':>7}  {'Basic':>7}  {'P4L Delta':>9}  {'Fixed?':>6}  Filename")
    print("  " + "-" * 88)

    fixed_count = 0
    improved_count = 0
    found_count = 0

    for i, r in enumerate(results):
        if not r["found"]:
            print(f"  {i+1:>3}  {r['old_avg']:>7.3f}  {'N/A':>7}  {'N/A':>7}  {'N/A':>9}  {'N/A':>6}  {r['filename']} [NOT FOUND]")
            continue

        found_count += 1
        delta = r["new_p4l_score"] - r["old_avg"]
        # Consider "fixed" if new score >= 0.5 (reasonable quality)
        is_fixed = r["new_p4l_score"] >= 0.5
        is_improved = delta > 0.05
        if is_fixed:
            fixed_count += 1
        if is_improved:
            improved_count += 1

        fix_label = "YES" if is_fixed else "no"
        print(f"  {i+1:>3}  {r['old_avg']:>7.3f}  {r['new_p4l_score']:>7.3f}  {r['new_basic_score']:>7.3f}  "
              f"{delta:>+9.3f}  {fix_label:>6}  {r['filename']}")

    # -----------------------------------------------------------------------
    # Phase 4: Conclusion
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    if found_count == 0:
        print("  No PDF files were found on disk -- cannot evaluate.")
    else:
        avg_old = sum(r["old_avg"] for r in results if r["found"]) / found_count
        avg_new_p4l = sum(r["new_p4l_score"] for r in results if r["found"]) / found_count
        avg_new_basic = sum(r["new_basic_score"] for r in results if r["found"]) / found_count

        print(f"  Files evaluated:  {found_count}")
        print(f"  Avg old indexed score:    {avg_old:.3f}")
        print(f"  Avg new pymupdf4llm score: {avg_new_p4l:.3f}")
        print(f"  Avg new basic pymupdf:     {avg_new_basic:.3f}")
        print(f"  Files improved (delta > 0.05): {improved_count}/{found_count}")
        print(f"  Files fixed (new score >= 0.5): {fixed_count}/{found_count}")
        print()

        if fixed_count == found_count:
            print("  VERDICT: pymupdf4llm 1.27.2.2 upgrade FIXES the garbage text problem.")
            print("  A separate quality scorer + OCR fallback is NOT needed.")
        elif fixed_count > found_count * 0.7:
            print("  VERDICT: pymupdf4llm 1.27.2.2 MOSTLY fixes the garbage text problem.")
            print(f"  {found_count - fixed_count} files still need quality-based OCR fallback.")
            print("  Recommend: upgrade + lightweight quality check for remaining edge cases.")
        elif improved_count > found_count * 0.5:
            print("  VERDICT: pymupdf4llm 1.27.2.2 IMPROVES but does NOT fix the problem.")
            print("  Many files still have poor quality. A quality scorer + OCR fallback is STILL NEEDED.")
        else:
            print("  VERDICT: pymupdf4llm 1.27.2.2 does NOT meaningfully fix the garbage text problem.")
            print("  A quality scorer + OCR fallback pipeline is STILL NEEDED.")

    print("\n" + "=" * 90)
    print("SPIKE V2 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
