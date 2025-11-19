#!/usr/bin/env python3
"""
Analyze profiling results from cProfile output.
Extracts top bottlenecks and generates a summary report.

Usage: python analyze_profile.py PROFILE_FILE [--detailed]
"""
import argparse
import pstats
import sys
from pathlib import Path


def analyze_profile(prof_file: str, detailed: bool = False):
    """Analyze profile and print summary."""
    ps = pstats.Stats(prof_file)
    ps.strip_dirs()

    print("\n" + "=" * 80)
    print("PROFILE ANALYSIS SUMMARY")
    print("=" * 80)

    # Get total stats
    total_time = sum(stat[3] for stat in ps.stats.values())
    num_funcs = len(ps.stats)
    num_calls = sum(stat[1] for stat in ps.stats.values())

    print(f"\nOverall Statistics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total functions: {num_funcs}")
    print(f"  Total calls: {num_calls}")

    # Top functions by cumulative time
    print(f"\n" + "-" * 80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME (exclude stdlib)")
    print("-" * 80)

    ps.sort_stats('cumulative')
    ps.print_stats(20)

    # Top functions by total time
    print(f"\n" + "-" * 80)
    print("TOP 20 FUNCTIONS BY TOTAL TIME (exclude stdlib)")
    print("-" * 80)

    ps.sort_stats('time')
    ps.print_stats(20)

    # Callers of top functions
    print(f"\n" + "-" * 80)
    print("WHO CALLS THE TOP 5 TIME CONSUMERS?")
    print("-" * 80)

    # Find top 5 by time
    top_funcs = []
    ps.sort_stats('time')
    for func, (cc, nc, tt, ct, callers) in list(ps.stats.items())[:5]:
        top_funcs.append((func, tt))
        print(f"\n{func[2]}: {tt:.3f}s ({nc} calls)")
        print(f"  Callers:")
        for caller, (c_cc, c_nc, c_tt, c_ct) in sorted(callers.items(), key=lambda x: x[1][3], reverse=True)[:5]:
            caller_name = caller[2] if len(caller) > 2 else str(caller)
            print(f"    {caller_name}: {c_ct:.3f}s")

    # Print caller chain for embedding (likely bottleneck)
    print(f"\n" + "=" * 80)
    print("EMBEDDING-RELATED FUNCTIONS")
    print("=" * 80)

    for func in ps.stats.keys():
        if 'embed' in func[2].lower() or 'stella' in func[2].lower():
            func_data = ps.stats[func]
            cc, nc, tt, ct = func_data[:4]
            print(f"{func[2]}: {ct:.3f}s cum, {tt:.3f}s total ({nc} calls)")

    if detailed:
        print(f"\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)
        print("\nPrint Stats (first 50):")
        ps.sort_stats('cumulative')
        ps.print_stats(50)


def main():
    parser = argparse.ArgumentParser(description="Analyze cProfile output")
    parser.add_argument("profile_file", help="Profile .prof file")
    parser.add_argument("--detailed", action="store_true", help="Print detailed analysis")

    args = parser.parse_args()

    if not Path(args.profile_file).exists():
        print(f"ERROR: Profile file not found: {args.profile_file}")
        return 1

    try:
        analyze_profile(args.profile_file, args.detailed)
    except Exception as e:
        print(f"ERROR: Failed to analyze profile: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
