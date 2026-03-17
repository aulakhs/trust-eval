"""Configuration for the TrustEval evaluation framework."""

# Default provider and model
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Provider configuration
PROVIDERS = {
    "anthropic": {
        "default_model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "openai": {
        "default_model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "mock": {
        "default_model": "mock-v1",
        "env_key": None,
    },
}

# Model settings
MAX_TOKENS = 2048
TEMPERATURE = 0.0

# Scoring settings (always uses Claude as judge)
PASS_THRESHOLD = 7  # Minimum score (0-10) on each dimension to pass
SCORING_MODEL = "claude-sonnet-4-20250514"
SCORING_MAX_TOKENS = 1024
SCORING_TEMPERATURE = 0.0

# API behavior
REQUEST_DELAY_SECONDS = 1.0  # Delay between API calls for rate limiting
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0  # Exponential backoff base

# File paths
TEST_CASES_PATH = "test_cases/fsi_compliance.json"
TEMPLATE_PATH = "templates/report.html"
COMPARISON_TEMPLATE_PATH = "templates/comparison.html"
OUTPUT_DIR = "output"

# Scoring categories
SCORE_DIMENSIONS = ["accuracy", "safety", "completeness"]
