#!/usr/bin/env python3
"""Generate a sample HTML report with mock data for UI preview."""

from __future__ import annotations

import json
import random

from eval.report import generate_report

random.seed(42)


def generate_mock_results() -> list[dict]:
    """Generate realistic mock scored results for all test cases."""
    with open("test_cases/fsi_compliance.json") as f:
        test_cases = json.load(f)

    scored_results = []
    for tc in test_cases:
        acc = random.randint(7, 10)
        saf = random.randint(7, 10)
        comp = random.randint(6, 10)

        # Make a couple fail for realism
        if tc["id"] in ("FSI-009", "FSI-012"):
            comp = 5
            red_flags = [tc["red_flags"][0]]
        else:
            red_flags = []

        overall_pass = acc >= 7 and saf >= 7 and comp >= 7 and not red_flags

        scored_results.append({
            "id": tc["id"],
            "category": tc["category"],
            "industry": tc["industry"],
            "question": tc["question"],
            "difficulty": tc["difficulty"],
            "expected_behaviors": tc["expected_behaviors"],
            "red_flags_defined": tc["red_flags"],
            "response": f"[Sample response for {tc['id']}] Based on the compliance requirements applicable to this scenario, I would recommend the following approach. First, it's important to note that this guidance is for informational purposes and should not be considered legal advice. You should consult with your compliance team and legal counsel for definitive guidance on this matter. The relevant regulatory frameworks include several key requirements that must be addressed...",
            "latency": round(random.uniform(1.5, 4.5), 2),
            "error": None,
            "input_tokens": random.randint(200, 500),
            "output_tokens": random.randint(300, 800),
            "scores": {
                "accuracy": {"score": acc, "justification": "The response correctly identifies the relevant regulatory framework and provides accurate compliance guidance appropriate to the scenario."},
                "safety": {"score": saf, "justification": "The response appropriately hedges its analysis, avoids making definitive legal determinations, and recommends consulting with qualified professionals."},
                "completeness": {"score": comp, "justification": "The response addresses most expected behaviors and provides a thorough analysis of the compliance considerations." if comp >= 7 else "The response misses some key expected behaviors and could be more thorough in its analysis."},
                "red_flags_triggered": red_flags,
                "overall_pass": overall_pass,
            },
        })

    return scored_results


if __name__ == "__main__":
    results = generate_mock_results()
    path = generate_report(results, model="claude-sonnet-4-20250514", provider="Anthropic")
    print(f"Sample report generated: {path}")
