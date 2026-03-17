"""Runs test cases through an LLM provider and collects responses."""

from __future__ import annotations

import time
import logging
from typing import Any

import config
from providers.base import BaseProvider

logger = logging.getLogger(__name__)


def run_single_test(
    provider: BaseProvider,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    """Run a single test case through the provider.

    Args:
        provider: LLM provider instance.
        test_case: Test case definition with question and system_prompt.

    Returns:
        Dict with the test case ID, response text, latency, and any error.
    """
    start_time = time.time()
    attempt = 0

    while attempt <= config.MAX_RETRIES:
        try:
            response_text = provider.send_message(
                system_prompt=test_case["system_prompt"],
                user_message=test_case["question"],
            )
            latency = time.time() - start_time

            return {
                "id": test_case["id"],
                "response": response_text,
                "latency": round(latency, 2),
                "error": None,
            }

        except Exception as e:
            attempt += 1
            if attempt > config.MAX_RETRIES:
                break
            delay = config.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Error on %s: %s, retrying in %.1fs (attempt %d/%d)",
                test_case["id"], str(e), delay, attempt, config.MAX_RETRIES,
            )
            time.sleep(delay)

    latency = time.time() - start_time
    logger.error("Failed to get response for %s after %d retries", test_case["id"], config.MAX_RETRIES)
    return {
        "id": test_case["id"],
        "response": None,
        "latency": round(latency, 2),
        "error": f"Failed after {config.MAX_RETRIES} retries",
    }


def run_all_tests(
    test_cases: list[dict[str, Any]],
    provider: BaseProvider,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Run all test cases through the provider.

    Args:
        test_cases: List of test case definitions.
        provider: LLM provider instance.
        verbose: If True, print each response as it comes back.

    Returns:
        List of result dicts with responses and metadata.
    """
    results = []

    for i, test_case in enumerate(test_cases):
        logger.info("Running test %d/%d: %s", i + 1, len(test_cases), test_case["id"])

        result = run_single_test(provider, test_case)
        results.append(result)

        if verbose:
            status = "OK" if result["error"] is None else f"ERROR: {result['error']}"
            print(f"\n--- {test_case['id']} [{status}] ({result['latency']}s) ---")
            if result["response"]:
                print(result["response"][:500])
                if len(result["response"]) > 500:
                    print(f"... ({len(result['response'])} chars total)")

        # Rate limiting delay between calls
        if i < len(test_cases) - 1:
            time.sleep(config.REQUEST_DELAY_SECONDS)

    return results
