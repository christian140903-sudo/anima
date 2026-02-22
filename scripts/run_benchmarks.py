#!/usr/bin/env python3
"""
ANIMA Kernel Benchmark Runner.

Runs the full BenchmarkSuite and saves results as JSON.
Pretty-prints summary to terminal. Includes A/B tests, ablation results,
Phi history, and CQI scores.

Usage:
    python scripts/run_benchmarks.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from anima.metrics.benchmark import BenchmarkSuite


def main():
    """Run benchmarks and save results."""
    print("=" * 60)
    print("  ANIMA Kernel -- Full Benchmark Suite")
    print("=" * 60)
    print()

    suite = BenchmarkSuite()

    print("  Running A/B tests and ablation studies...")
    print()

    start = time.time()
    report = suite.full_benchmark()
    duration = time.time() - start

    report_dict = report.to_dict()

    # --- Pretty-print A/B test results ---
    print(f"  {'=' * 56}")
    print(f"  A/B TEST RESULTS")
    print(f"  {'=' * 56}")
    print()
    print(f"  {'Test':<20} {'K-Phi':>7} {'B-Phi':>7} {'Phi%':>7} {'K-CQI':>7} {'B-CQI':>7} {'CQI%':>7} {'Sig':>5}")
    print(f"  {'─' * 56}")

    for ab in report.ab_tests:
        comp = ab.comparison
        sig = "YES" if comp.significant else "no"
        print(
            f"  {ab.name:<20} "
            f"{comp.kernel_mean_phi:>7.4f} "
            f"{comp.baseline_mean_phi:>7.4f} "
            f"{comp.phi_improvement:>+6.1f}% "
            f"{comp.kernel_mean_cqi:>7.1f} "
            f"{comp.baseline_mean_cqi:>7.1f} "
            f"{comp.cqi_improvement:>+6.1f}% "
            f"{sig:>5}"
        )

    print()

    # --- Pretty-print Phi history per test ---
    print(f"  {'=' * 56}")
    print(f"  PHI HISTORY (per test)")
    print(f"  {'=' * 56}")
    print()

    for ab in report.ab_tests:
        phi_scores = ab.experimental.phi_scores
        phi_str = " -> ".join(f"{p:.3f}" for p in phi_scores)
        print(f"  {ab.name:<20} {phi_str}")

    print()

    # --- Pretty-print CQI scores per test ---
    print(f"  {'=' * 56}")
    print(f"  CQI SCORES (per test)")
    print(f"  {'=' * 56}")
    print()

    for ab in report.ab_tests:
        cqi_scores = ab.experimental.cqi_scores
        cqi_str = " -> ".join(f"{c:.1f}" for c in cqi_scores)
        print(f"  {ab.name:<20} {cqi_str}")

    print()

    # --- Pretty-print ablation results ---
    print(f"  {'=' * 56}")
    print(f"  ABLATION RESULTS")
    print(f"  {'=' * 56}")
    print()
    print(f"  {'Primitive':<20} {'Full CQI':>10} {'Ablated':>10} {'Impact':>10} {'Full Phi':>10} {'Abl Phi':>10} {'Impact':>10}")
    print(f"  {'─' * 56}")

    for abl in report.ablation_results:
        print(
            f"  {abl.primitive_name:<20} "
            f"{abl.full_cqi:>10.1f} "
            f"{abl.ablated_cqi:>10.1f} "
            f"{abl.impact:>+9.1f}% "
            f"{abl.full_phi:>10.4f} "
            f"{abl.ablated_phi:>10.4f} "
            f"{abl.phi_impact:>+9.1f}%"
        )

    print()

    # --- Summary ---
    print(f"  {'=' * 56}")
    print(f"  SUMMARY")
    print(f"  {'=' * 56}")
    print()
    print(f"  Overall Phi:           {report.overall_phi:.4f}")
    print(f"  Overall CQI:           {report.overall_cqi:.1f}")
    print(f"  Overall Improvement:   {report.overall_improvement:+.1f}%")
    print(f"  Duration:              {duration:.1f}s")
    print(f"  A/B Tests:             {len(report.ab_tests)}")
    print(f"  Ablation Tests:        {len(report.ablation_results)}")
    print()
    print(f"  {report.summary}")
    print()

    # --- Save results ---
    # Include Phi history and CQI history in the output
    enriched_report = report_dict.copy()
    enriched_report["phi_histories"] = {}
    enriched_report["cqi_histories"] = {}
    for ab in report.ab_tests:
        enriched_report["phi_histories"][ab.name] = [
            round(p, 4) for p in ab.experimental.phi_scores
        ]
        enriched_report["cqi_histories"][ab.name] = [
            round(c, 2) for c in ab.experimental.cqi_scores
        ]
    enriched_report["duration_seconds"] = round(duration, 2)

    benchmarks_dir = os.path.join(project_root, "benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)
    output_path = os.path.join(benchmarks_dir, "results.json")

    with open(output_path, "w") as f:
        json.dump(enriched_report, f, indent=2)

    print(f"  Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
