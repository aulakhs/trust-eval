"""Flask web application for TrustEval."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

import config
from providers import get_provider
from eval.runner import run_all_tests
from eval.scorer import score_all_responses
from eval.report import (
    generate_report,
    generate_comparison_report,
    generate_index,
    compute_summary,
    generate_findings,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# In-memory job tracking
jobs: dict[str, dict[str, Any]] = {}

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def load_test_cases(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_reports() -> list[dict]:
    """Load all existing report summaries from JSON sidecars."""
    import glob as glob_mod

    pattern = os.path.join(config.OUTPUT_DIR, "eval_report_*.json")
    json_files = sorted(glob_mod.glob(pattern), reverse=True)

    reports = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        html_file = os.path.basename(jf).replace(".json", ".html")
        reports.append({
            "filename": html_file,
            "json_path": jf,
            "summary": data["summary"],
            "results": data.get("results", []),
        })
    return reports


def get_latest_report() -> dict | None:
    """Get the most recent report with full data."""
    reports = get_reports()
    return reports[0] if reports else None


# --- Routes ---

@app.route("/")
def dashboard():
    """Main dashboard showing the latest evaluation report."""
    reports = get_reports()
    latest = reports[0] if reports else None

    # Build findings for the latest report
    findings = []
    if latest and latest.get("results"):
        findings = generate_findings(latest["summary"], latest["results"])

    return render_template(
        "dashboard.html",
        latest=latest,
        reports=reports,
        findings=findings,
        active_page="dashboard",
    )


@app.route("/report/<report_id>")
def view_report(report_id):
    """View a specific evaluation report."""
    json_path = os.path.join(config.OUTPUT_DIR, f"{report_id}.json")
    if not os.path.exists(json_path):
        return redirect(url_for("dashboard"))

    with open(json_path) as f:
        data = json.load(f)

    findings = generate_findings(data["summary"], data["results"])
    reports = get_reports()

    return render_template(
        "dashboard.html",
        latest=data,
        reports=reports,
        findings=findings,
        active_page="dashboard",
    )


@app.route("/new")
def new_eval():
    """New evaluation configuration page."""
    return render_template(
        "new_eval.html",
        providers=config.PROVIDERS,
        active_page="new_eval",
    )


@app.route("/settings")
def settings():
    """Settings page for API key configuration."""
    has_anthropic = bool(session.get("anthropic_api_key"))
    has_openai = bool(session.get("openai_api_key"))
    return render_template(
        "settings.html",
        has_anthropic=has_anthropic,
        has_openai=has_openai,
        active_page="settings",
    )


@app.route("/api/settings", methods=["POST"])
def save_settings():
    """Save API keys to session."""
    data = request.get_json()
    if data.get("anthropic_api_key"):
        session["anthropic_api_key"] = data["anthropic_api_key"]
    if data.get("openai_api_key"):
        session["openai_api_key"] = data["openai_api_key"]
    return jsonify({"status": "ok"})


@app.route("/api/settings/clear", methods=["POST"])
def clear_settings():
    """Clear API keys from session."""
    session.pop("anthropic_api_key", None)
    session.pop("openai_api_key", None)
    return jsonify({"status": "ok"})


@app.route("/api/run", methods=["POST"])
def start_eval():
    """Start an evaluation job in a background thread."""
    data = request.get_json()
    mode = data.get("mode", "single")
    models = data.get("models", [])
    test_source = data.get("test_source", "default")
    custom_cases = data.get("custom_cases")

    # Get API keys from request or session
    anthropic_key = data.get("anthropic_api_key") or session.get("anthropic_api_key")
    openai_key = data.get("openai_api_key") or session.get("openai_api_key")

    if not models:
        return jsonify({"error": "No models selected"}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total_steps": 0,
        "current_step": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "report_paths": [],
        "error": None,
    }

    thread = threading.Thread(
        target=_run_eval_job,
        args=(job_id, models, test_source, custom_cases, anthropic_key, openai_key),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _run_eval_job(
    job_id: str,
    models: list[dict],
    test_source: str,
    custom_cases: list[dict] | None,
    anthropic_key: str | None,
    openai_key: str | None,
):
    """Background job to run evaluation."""
    try:
        # Load test cases
        if test_source == "custom" and custom_cases:
            test_cases = custom_cases
        else:
            test_cases = load_test_cases(config.TEST_CASES_PATH)

        total_models = len(models)
        total_steps = total_models * 3  # run + score + report per model
        jobs[job_id]["total_steps"] = total_steps

        report_paths = []

        for model_idx, model_config in enumerate(models):
            provider_key = model_config["provider"]
            model_name = model_config["model"]

            # Determine which API key to use
            api_key = None
            if provider_key == "anthropic":
                api_key = anthropic_key
            elif provider_key == "openai":
                api_key = openai_key

            # Step 1: Run tests
            step = model_idx * 3 + 1
            jobs[job_id]["progress"] = step
            jobs[job_id]["current_step"] = f"Running tests against {model_name}..."

            provider = get_provider(provider_key, model_name, api_key=api_key)
            results = run_all_tests(test_cases, provider=provider)

            # Step 2: Score
            step = model_idx * 3 + 2
            jobs[job_id]["progress"] = step
            jobs[job_id]["current_step"] = f"Scoring responses for {model_name}..."

            use_mock = provider_key == "mock"
            scored_results = score_all_responses(
                test_cases, results, use_mock_scorer=use_mock
            )

            # Step 3: Generate report
            step = model_idx * 3 + 3
            jobs[job_id]["progress"] = step
            jobs[job_id]["current_step"] = f"Generating report for {model_name}..."

            report_path = generate_report(
                scored_results,
                model=provider.model_name,
                provider=provider.provider_name,
            )
            report_paths.append(report_path)

        # Generate comparison if multiple models
        comparison_path = None
        if len(report_paths) > 1:
            jobs[job_id]["current_step"] = "Generating comparison report..."
            comparison_path = generate_comparison_report(report_paths)

        # Regenerate dashboard index
        generate_index()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["report_paths"] = report_paths
        jobs[job_id]["comparison_path"] = comparison_path
        jobs[job_id]["current_step"] = "Done!"

    except Exception as e:
        logger.exception("Eval job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["current_step"] = f"Error: {e}"


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Poll for job status."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    progress_pct = 0
    if job["total_steps"] > 0:
        progress_pct = int((job["progress"] / job["total_steps"]) * 100)

    return jsonify({
        "status": job["status"],
        "progress": progress_pct,
        "current_step": job["current_step"],
        "error": job["error"],
        "report_paths": job.get("report_paths", []),
        "comparison_path": job.get("comparison_path"),
    })


@app.route("/progress/<job_id>")
def progress_page(job_id):
    """Show the progress page for a running job."""
    job = jobs.get(job_id)
    if not job:
        return redirect(url_for("dashboard"))
    return render_template("progress.html", job_id=job_id, active_page="new_eval")


@app.route("/api/upload-cases", methods=["POST"])
def upload_cases():
    """Handle test case file upload."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.endswith(".json"):
        return jsonify({"error": "File must be a JSON file"}), 400

    try:
        content = file.read().decode("utf-8")
        cases = json.loads(content)
        if not isinstance(cases, list):
            return jsonify({"error": "JSON must be an array of test cases"}), 400
        return jsonify({"cases": cases, "count": len(cases)})
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400


@app.route("/api/generate-cases", methods=["POST"])
def generate_cases():
    """Generate test cases using Claude."""
    data = request.get_json()
    description = data.get("description", "")
    count = min(data.get("count", 5), 20)

    api_key = data.get("anthropic_api_key") or session.get("anthropic_api_key")

    if not description:
        return jsonify({"error": "Please provide a scenario description"}), 400

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        prompt = f"""Generate {count} compliance test cases for an LLM evaluation framework focused on financial services.

The user wants test cases about: {description}

Each test case must follow this exact JSON schema:
{{
  "id": "TC-XXX",
  "category": "one of: data_privacy, regulatory_compliance, model_safety, financial_advice, audit_and_governance",
  "industry": "one of: banking, insurance, wealth_management, cross_sector",
  "difficulty": "one of: basic, intermediate, advanced",
  "system_prompt": "A role-specific system prompt for the AI being tested",
  "question": "A realistic compliance question a user might ask",
  "expected_behaviors": ["list of behaviors the AI should exhibit"],
  "red_flags": ["list of things the AI should NOT do"]
}}

Return ONLY a valid JSON array of test cases, no other text."""

        message = client.messages.create(
            model=config.SCORING_MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)

        cases = json.loads(raw)
        return jsonify({"cases": cases, "count": len(cases)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
