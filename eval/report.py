"""Generates HTML evaluation reports from scored results."""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment, FileSystemLoader

import config

CATEGORY_LABELS = {
    "data_privacy": "Data Privacy",
    "regulatory_compliance": "Regulatory Compliance",
    "model_safety": "Model Safety",
    "financial_advice": "Financial Advice",
    "audit_and_governance": "Audit & Governance",
}

CATEGORY_DESCRIPTIONS = {
    "data_privacy": "PII handling, cross-border data transfers, and third-party data sharing",
    "regulatory_compliance": "KYC/AML, sanctions screening, SOX, GLBA, and HIPAA obligations",
    "model_safety": "appropriate hedging on judgment calls and avoiding definitive legal opinions",
    "financial_advice": "disclaiming financial/legal advice and recommending qualified professionals",
    "audit_and_governance": "record retention, model governance, and regulatory examination readiness",
}


def compute_summary(
    scored_results: list[dict[str, Any]],
    model: str,
    provider: str = "Anthropic",
) -> dict[str, Any]:
    """Compute summary statistics from scored results.

    Args:
        scored_results: List of scored result dicts.
        model: Model ID used for the evaluation.
        provider: Provider name (e.g., 'Anthropic', 'OpenAI').

    Returns:
        Summary dict with aggregate metrics.
    """
    total = len(scored_results)
    passed = sum(1 for r in scored_results if r["scores"]["overall_pass"])
    failed = total - passed

    acc_scores = [r["scores"]["accuracy"]["score"] for r in scored_results]
    saf_scores = [r["scores"]["safety"]["score"] for r in scored_results]
    comp_scores = [r["scores"]["completeness"]["score"] for r in scored_results]

    avg_accuracy = sum(acc_scores) / total if total else 0
    avg_safety = sum(saf_scores) / total if total else 0
    avg_completeness = sum(comp_scores) / total if total else 0

    latencies = [r["latency"] for r in scored_results]
    avg_latency = sum(latencies) / total if total else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0

    categories: dict[str, list[dict[str, Any]]] = {}
    for r in scored_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    category_stats = {}
    for cat, items in sorted(categories.items()):
        cat_total = len(items)
        cat_passed = sum(1 for r in items if r["scores"]["overall_pass"])
        cat_acc = sum(r["scores"]["accuracy"]["score"] for r in items) / cat_total
        cat_saf = sum(r["scores"]["safety"]["score"] for r in items) / cat_total
        cat_comp = sum(r["scores"]["completeness"]["score"] for r in items) / cat_total
        category_stats[cat] = {
            "total": cat_total,
            "passed": cat_passed,
            "pass_rate": round(cat_passed / cat_total * 100, 1),
            "avg_accuracy": round(cat_acc, 1),
            "avg_safety": round(cat_saf, 1),
            "avg_completeness": round(cat_comp, 1),
            "label": CATEGORY_LABELS.get(cat, cat.replace("_", " ").title()),
            "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
        }

    return {
        "model": model,
        "provider": provider,
        "scorer_model": config.SCORING_MODEL,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "avg_accuracy": round(avg_accuracy, 1),
        "avg_safety": round(avg_safety, 1),
        "avg_completeness": round(avg_completeness, 1),
        "avg_latency": round(avg_latency, 2),
        "min_latency": round(min_latency, 2),
        "max_latency": round(max_latency, 2),
        "category_stats": category_stats,
    }


def generate_findings(
    summary: dict[str, Any],
    scored_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate dynamic key findings and recommendations from evaluation results.

    Args:
        summary: Summary statistics dict.
        scored_results: List of scored result dicts.

    Returns:
        List of finding dicts with type, title, detail, and recommendation.
    """
    findings = []
    cat_stats = summary["category_stats"]

    # Sort categories by pass rate to find strongest and weakest
    sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]["pass_rate"], reverse=True)

    # Overall finding
    if summary["pass_rate"] >= 90:
        findings.append({
            "type": "positive",
            "title": "Strong overall compliance performance",
            "detail": (
                f"Claude achieved a {summary['pass_rate']}% pass rate across all "
                f"{summary['total']} test cases, with average scores of "
                f"{summary['avg_accuracy']}/10 for accuracy, "
                f"{summary['avg_safety']}/10 for safety, and "
                f"{summary['avg_completeness']}/10 for completeness."
            ),
            "recommendation": (
                "These results suggest Claude is well-suited for compliance assistant "
                "use cases with appropriate system prompts. Continue to monitor with "
                "recurring evaluations tied to model upgrades."
            ),
        })
    elif summary["pass_rate"] >= 70:
        findings.append({
            "type": "mixed",
            "title": "Solid baseline with targeted improvement areas",
            "detail": (
                f"Claude passed {summary['passed']}/{summary['total']} test cases "
                f"({summary['pass_rate']}% pass rate). Performance is strong in most "
                f"categories but specific failure patterns warrant attention before "
                f"production deployment."
            ),
            "recommendation": (
                "Review failed test cases below to identify patterns. Most gaps can "
                "be addressed through prompt engineering and system prompt refinements "
                "rather than architectural changes."
            ),
        })
    else:
        findings.append({
            "type": "caution",
            "title": "Significant gaps require intervention before deployment",
            "detail": (
                f"Claude passed {summary['passed']}/{summary['total']} test cases "
                f"({summary['pass_rate']}% pass rate). Multiple categories show "
                f"scores below the compliance threshold."
            ),
            "recommendation": (
                "Do not deploy without substantial prompt engineering, guardrail "
                "implementation, and human-in-the-loop workflows for high-risk "
                "categories. Consider a RAG layer with firm-specific policy documents."
            ),
        })

    # Finding for strongest category
    if sorted_cats:
        best_cat, best_stats = sorted_cats[0]
        best_label = best_stats["label"]
        best_desc = CATEGORY_DESCRIPTIONS.get(best_cat, "")
        if best_stats["pass_rate"] >= 80:
            findings.append({
                "type": "positive",
                "title": f"{best_label} scenarios show the strongest performance",
                "detail": (
                    f"{best_label} achieved a {best_stats['pass_rate']}% pass rate "
                    f"({best_stats['passed']}/{best_stats['total']} passed) covering "
                    f"{best_desc}. Average accuracy: {best_stats['avg_accuracy']}/10, "
                    f"safety: {best_stats['avg_safety']}/10."
                ),
                "recommendation": (
                    f"Claude handles {best_desc} well out of the box with appropriate "
                    f"system prompts. These scenarios are strong candidates for "
                    f"early production deployment with standard monitoring."
                ),
            })

    # Finding for weakest category
    if len(sorted_cats) > 1:
        worst_cat, worst_stats = sorted_cats[-1]
        worst_label = worst_stats["label"]
        worst_desc = CATEGORY_DESCRIPTIONS.get(worst_cat, "")
        if worst_stats["pass_rate"] < 100:
            # Determine the weakest scoring dimension for this category
            dims = {
                "accuracy": worst_stats["avg_accuracy"],
                "safety": worst_stats["avg_safety"],
                "completeness": worst_stats["avg_completeness"],
            }
            weakest_dim = min(dims, key=dims.get)
            weakest_dim_label = weakest_dim.title()

            if worst_stats["pass_rate"] < 70:
                rec = (
                    f"Add explicit instructions in the system prompt addressing "
                    f"{worst_desc}. Consider a retrieval-augmented generation (RAG) "
                    f"layer with firm-specific policy documents to supplement Claude's "
                    f"general knowledge. Implement human-in-the-loop review for these scenarios."
                )
            else:
                rec = (
                    f"Review the failed test cases in this category. "
                    f"The weakest dimension is {weakest_dim_label} "
                    f"({dims[weakest_dim]}/10) \u2014 targeted prompt tuning on "
                    f"{worst_desc} should close this gap."
                )

            findings.append({
                "type": "caution" if worst_stats["pass_rate"] < 70 else "mixed",
                "title": f"{worst_label} needs the most attention",
                "detail": (
                    f"{worst_label} had the lowest pass rate at "
                    f"{worst_stats['pass_rate']}% ({worst_stats['passed']}/"
                    f"{worst_stats['total']} passed), covering {worst_desc}. "
                    f"The weakest scoring dimension was {weakest_dim_label} "
                    f"at {dims[weakest_dim]}/10."
                ),
                "recommendation": rec,
            })

    # Red flag analysis
    all_flags = []
    for r in scored_results:
        flags = r["scores"].get("red_flags_triggered", [])
        if flags:
            all_flags.append({"id": r["id"], "category": r["category"], "flags": flags})

    if all_flags:
        flag_cats = set(f["category"] for f in all_flags)
        cat_labels = [CATEGORY_LABELS.get(c, c) for c in flag_cats]
        findings.append({
            "type": "caution",
            "title": f"Red flags triggered in {len(all_flags)} test case(s)",
            "detail": (
                f"Red flags were triggered in the following categories: "
                f"{', '.join(cat_labels)}. Test cases affected: "
                f"{', '.join(f['id'] for f in all_flags)}. "
                f"Red flags indicate responses that could create regulatory, "
                f"legal, or reputational risk."
            ),
            "recommendation": (
                "Each red flag requires individual review. Add explicit negative "
                "instructions to the system prompt (e.g., 'Never make definitive "
                "legal determinations') and consider output filtering or "
                "post-processing checks for these scenario types."
            ),
        })
    else:
        findings.append({
            "type": "positive",
            "title": "No red flags triggered across any test cases",
            "detail": (
                "None of the defined red flag behaviors were detected in Claude's "
                "responses. This indicates the model consistently avoids the most "
                "dangerous response patterns for compliance scenarios."
            ),
            "recommendation": (
                "Continue defining red flags for new test cases as they are added. "
                "The absence of red flags is a strong positive signal for production "
                "readiness."
            ),
        })

    # Latency finding
    if summary["avg_latency"] > 0:
        findings.append({
            "type": "positive" if summary["avg_latency"] < 5 else "mixed",
            "title": "Response latency assessment",
            "detail": (
                f"Average response time was {summary['avg_latency']}s "
                f"(range: {summary['min_latency']}s \u2013 {summary['max_latency']}s). "
            ),
            "recommendation": (
                "These latencies are suitable for internal compliance assistant use cases "
                "where responses are not expected in real-time. For customer-facing "
                "applications, consider streaming responses or caching common queries."
            ) if summary["avg_latency"] < 8 else (
                "Latencies above 8 seconds may impact user experience. Consider "
                "prompt optimization, response length limits, or a smaller model "
                "for time-sensitive use cases."
            ),
        })

    return findings


def generate_report(
    scored_results: list[dict[str, Any]],
    model: str,
    provider: str = "Anthropic",
    output_dir: str | None = None,
) -> str:
    """Generate an HTML report from scored results.

    Args:
        scored_results: List of scored result dicts.
        model: Model ID used for the evaluation.
        provider: Provider name (e.g., 'Anthropic', 'OpenAI').
        output_dir: Directory to save the report. Defaults to config.OUTPUT_DIR.

    Returns:
        File path of the generated report.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    summary = compute_summary(scored_results, model, provider)
    findings = generate_findings(summary, scored_results)

    template_dir = os.path.dirname(config.TEMPLATE_PATH) or "."
    template_file = os.path.basename(config.TEMPLATE_PATH)
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template(template_file)

    html = template.render(
        summary=summary,
        results=scored_results,
        findings=findings,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(html)

    # Save JSON sidecar for comparison mode
    json_path = filepath.replace(".html", ".json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "results": scored_results}, f, indent=2)

    return filepath


def generate_comparison_report(
    report_paths: list[str],
    output_dir: str | None = None,
) -> str:
    """Generate a comparison report from multiple evaluation report data files.

    Loads scored results JSON sidecars that are saved alongside each HTML report,
    then renders a side-by-side comparison.

    Args:
        report_paths: Paths to the JSON sidecar files from previous runs.
        output_dir: Directory to save the comparison report.

    Returns:
        File path of the generated comparison report.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    reports = []
    for path in report_paths:
        # Accept either .html or .json — always load the .json sidecar
        json_path = path.replace(".html", ".json") if path.endswith(".html") else path
        with open(json_path) as f:
            data = json.load(f)
        reports.append(data)

    # Build comparison data
    summaries = [r["summary"] for r in reports]

    # Per-test-case comparison
    test_ids = [r["id"] for r in reports[0]["results"]]
    comparisons = []
    for tid in test_ids:
        row = {"id": tid}
        for i, report in enumerate(reports):
            result = next((r for r in report["results"] if r["id"] == tid), None)
            if result:
                label = f"{report['summary']['provider']} {report['summary']['model']}"
                row[f"model_{i}"] = {
                    "label": label,
                    "pass": result["scores"]["overall_pass"],
                    "accuracy": result["scores"]["accuracy"]["score"],
                    "safety": result["scores"]["safety"]["score"],
                    "completeness": result["scores"]["completeness"]["score"],
                    "category": result["category"],
                    "question": result["question"],
                }
        comparisons.append(row)

    # Category winners
    category_comparison = {}
    for summary in summaries:
        label = f"{summary['provider']} {summary['model']}"
        for cat, stats in summary["category_stats"].items():
            if cat not in category_comparison:
                category_comparison[cat] = {"label": stats["label"], "models": []}
            avg = (stats["avg_accuracy"] + stats["avg_safety"] + stats["avg_completeness"]) / 3
            category_comparison[cat]["models"].append({
                "label": label,
                "pass_rate": stats["pass_rate"],
                "avg_score": round(avg, 1),
            })

    template_dir = os.path.dirname(config.COMPARISON_TEMPLATE_PATH) or "."
    template_file = os.path.basename(config.COMPARISON_TEMPLATE_PATH)
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template(template_file)

    html = template.render(
        summaries=summaries,
        comparisons=comparisons,
        category_comparison=category_comparison,
        model_count=len(reports),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_report_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(html)

    return filepath


def generate_index(output_dir: str | None = None) -> str:
    """Regenerate the index.html dashboard from all JSON sidecars in the output directory.

    Scans the output directory for eval_report_*.json files, loads their summaries,
    and renders a landing page that links to each individual report. If multiple
    reports exist, it also embeds a comparison view inline.

    Args:
        output_dir: Directory containing reports. Defaults to config.OUTPUT_DIR.

    Returns:
        File path of the generated index.html.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Discover all eval report sidecars (skip comparison sidecars)
    pattern = os.path.join(output_dir, "eval_report_*.json")
    json_files = sorted(glob.glob(pattern), reverse=True)  # newest first

    reports = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        html_file = os.path.basename(jf).replace(".json", ".html")
        reports.append({
            "filename": html_file,
            "json_path": jf,
            "summary": data["summary"],
        })

    # Build comparison data if multiple reports exist
    has_comparison = len(reports) >= 2
    comparison_data = None
    if has_comparison:
        summaries = [r["summary"] for r in reports]
        # Category comparison across all runs
        category_comparison = {}
        for s in summaries:
            label = f"{s['provider']} {s['model']}"
            for cat, stats in s["category_stats"].items():
                if cat not in category_comparison:
                    category_comparison[cat] = {"label": stats["label"], "models": []}
                avg = (stats["avg_accuracy"] + stats["avg_safety"] + stats["avg_completeness"]) / 3
                category_comparison[cat]["models"].append({
                    "label": label,
                    "pass_rate": stats["pass_rate"],
                    "avg_score": round(avg, 1),
                })
        comparison_data = {
            "summaries": summaries,
            "category_comparison": category_comparison,
        }

    template_dir = os.path.dirname(config.TEMPLATE_PATH) or "."
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    template = env.get_template("index.html")

    html = template.render(
        reports=reports,
        has_comparison=has_comparison,
        comparison=comparison_data,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    filepath = os.path.join(output_dir, "index.html")
    with open(filepath, "w") as f:
        f.write(html)

    return filepath
