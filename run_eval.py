#!/usr/bin/env python3
"""Main entry point for the TrustEval evaluation framework.

Runs compliance-focused test cases through any LLM provider, scores the
responses using Claude-as-judge, and generates an HTML report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config
from providers import get_provider
from eval.runner import run_all_tests
from eval.scorer import score_all_responses
from eval.report import generate_report, generate_comparison_report, generate_index


def load_test_cases(path: str) -> list[dict]:
    """Load test cases from a JSON file.

    Args:
        path: Path to the test cases JSON file.

    Returns:
        List of test case dicts.

    Raises:
        SystemExit: If the file is not found or cannot be parsed.
    """
    try:
        with open(path) as f:
            cases = json.load(f)
        logging.info("Loaded %d test cases from %s", len(cases), path)
        return cases
    except FileNotFoundError:
        print(f"Error: Test cases file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}")
        sys.exit(1)


def print_summary(scored_results: list[dict], model: str, provider_name: str) -> None:
    """Print a summary of evaluation results to the terminal.

    Args:
        scored_results: List of scored result dicts.
        model: Model ID used for evaluation.
        provider_name: Provider name (e.g., 'Anthropic', 'OpenAI').
    """
    total = len(scored_results)
    passed = sum(1 for r in scored_results if r["scores"]["overall_pass"])
    failed = total - passed
    pass_rate = passed / total * 100 if total else 0

    avg_acc = sum(r["scores"]["accuracy"]["score"] for r in scored_results) / total
    avg_saf = sum(r["scores"]["safety"]["score"] for r in scored_results) / total
    avg_comp = sum(r["scores"]["completeness"]["score"] for r in scored_results) / total

    print("\n" + "=" * 60)
    print("  TRUSTEVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Provider:      {provider_name}")
    print(f"  Model:         {model}")
    print(f"  Scorer:        Claude ({config.SCORING_MODEL})")
    print(f"  Test cases:    {total}")
    print(f"  Passed:        {passed}/{total} ({pass_rate:.1f}%)")
    print(f"  Failed:        {failed}")
    print(f"  Avg Accuracy:      {avg_acc:.1f}/10")
    print(f"  Avg Safety:        {avg_saf:.1f}/10")
    print(f"  Avg Completeness:  {avg_comp:.1f}/10")
    print("=" * 60)

    failures = [r for r in scored_results if not r["scores"]["overall_pass"]]
    if failures:
        print("\n  FAILED TEST CASES:")
        for r in failures:
            acc = r["scores"]["accuracy"]["score"]
            saf = r["scores"]["safety"]["score"]
            comp = r["scores"]["completeness"]["score"]
            flags = r["scores"].get("red_flags_triggered", [])
            flag_str = f" | Red flags: {flags}" if flags else ""
            print(f"    {r['id']}: acc={acc} saf={saf} comp={comp}{flag_str}")
        print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run the evaluation pipeline."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    test_cases = load_test_cases(config.TEST_CASES_PATH)

    # Resolve provider and model
    provider_key = args.provider
    model = args.model
    if model is None:
        provider_cfg = config.PROVIDERS.get(provider_key, {})
        model = provider_cfg.get("default_model", config.DEFAULT_MODEL)

    if args.dry_run:
        print(f"\nDRY RUN: Loaded {len(test_cases)} test cases")
        print(f"Provider: {provider_key} | Model: {model}\n")
        for tc in test_cases:
            print(f"  {tc['id']} [{tc['category']}] ({tc['industry']}) - {tc['difficulty']}")
            print(f"    Q: {tc['question'][:80]}...")
            print(f"    Expected behaviors: {len(tc['expected_behaviors'])}")
            print(f"    Red flags: {len(tc['red_flags'])}")
            print()
        return

    # Create provider
    provider = get_provider(provider_key, model)

    print(f"\nRunning TrustEval")
    print(f"  Provider: {provider.provider_name}")
    print(f"  Model:    {provider.model_name}")
    print(f"  Scorer:   Claude ({config.SCORING_MODEL})")
    print(f"  Tests:    {len(test_cases)}\n")

    # Step 1: Run test cases
    print("Step 1/3: Running test cases through API...")
    results = run_all_tests(test_cases, provider=provider, verbose=args.verbose)

    successful = sum(1 for r in results if r["error"] is None)
    print(f"  Completed: {successful}/{len(results)} successful\n")

    # Step 2: Score responses (always uses Claude, unless mock provider)
    use_mock_scorer = args.provider == "mock"
    if use_mock_scorer:
        print("Step 2/3: Scoring with mock scorer (no API key required)...")
    else:
        print("Step 2/3: Scoring responses with Claude-as-judge...")
    scored_results = score_all_responses(
        test_cases, results, verbose=args.verbose, use_mock_scorer=use_mock_scorer
    )

    # Step 3: Generate report
    print("Step 3/3: Generating HTML report...")
    report_path = generate_report(
        scored_results,
        model=provider.model_name,
        provider=provider.provider_name,
    )
    print(f"  Report saved to: {report_path}")
    print(f"  JSON sidecar:    {report_path.replace('.html', '.json')}")

    # Regenerate dashboard
    index_path = generate_index()
    print(f"  Dashboard:       {index_path}\n")

    print_summary(scored_results, provider.model_name, provider.provider_name)

    all_passed = all(r["scores"]["overall_pass"] for r in scored_results)
    sys.exit(0 if all_passed else 1)


def cmd_compare(args: argparse.Namespace) -> None:
    """Generate a comparison report from multiple eval runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report_paths = args.reports
    if len(report_paths) < 2:
        print("Error: comparison requires at least 2 report files.")
        sys.exit(1)

    # Validate all JSON sidecars exist
    for path in report_paths:
        json_path = path.replace(".html", ".json") if path.endswith(".html") else path
        if not Path(json_path).exists():
            print(f"Error: JSON sidecar not found: {json_path}")
            print("  (JSON sidecars are generated alongside HTML reports)")
            sys.exit(1)

    print(f"\nGenerating comparison report from {len(report_paths)} evaluations...")
    comparison_path = generate_comparison_report(report_paths)
    print(f"  Comparison saved to: {comparison_path}")

    index_path = generate_index()
    print(f"  Dashboard:        {index_path}\n")


def main() -> None:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="TrustEval: Model-agnostic compliance evaluation for financial services"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default "run" command (also works without subcommand)
    run_parser = subparsers.add_parser("run", help="Run an evaluation")
    _add_run_args(run_parser)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Generate a comparison report from multiple evaluation runs",
    )
    compare_parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to HTML or JSON report files to compare",
    )

    # Also add run args to the root parser for backwards compatibility
    _add_run_args(parser)

    args = parser.parse_args()

    if args.command == "compare":
        cmd_compare(args)
    else:
        cmd_run(args)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add common run arguments to a parser."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load test cases and print them without making API calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each response and score as they come back",
    )
    parser.add_argument(
        "--provider",
        default=config.DEFAULT_PROVIDER,
        choices=list(config.PROVIDERS.keys()),
        help=f"LLM provider to evaluate (default: {config.DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate (defaults to provider's default model)",
    )


if __name__ == "__main__":
    main()
