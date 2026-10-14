from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


ALLOWED_SUPPORT_LEVELS = {"strong", "moderate", "weak"}
PROMPT_PLACEHOLDERS = {
    "{{FEATURE_DICTIONARY}}": "feature dictionary",
    "{{PHQ9_DICTIONARY}}": "PHQ-9 dictionary",
    "{{CASE_JSON}}": "case JSON",
}
DEFAULT_ENV_FILE = Path(".env")
ENV_FILE_CANDIDATES = [
    Path(".env"),
    Path("experimento_llm/.env"),
    Path("experimento_llm/.venv/.env"),
]
QWEN_KEY_CANDIDATES = [
    "OPENROUTER_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
]
ENV_PROVIDER_NAMES = {"openai", "gemini", "claude", "qwen"}
FORBIDDEN_CASE_KEYS = {
    "uid",
    "type",
    "date",
    "reference_date",
    "window_start",
    "window_end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the qualitative PHQ-9 and behavioral-pattern LLM experiment "
            "with independent API calls for each StudentLife case."
        )
    )
    parser.add_argument(
        "--prompt",
        default=Path("prompts/qualitative_prompt_v1.md"),
        type=Path,
        help="Prompt template used for every case.",
    )
    parser.add_argument(
        "--feature-dictionary",
        default=Path("outputs/llm_feature_dictionary.json"),
        type=Path,
        help="JSON feature dictionary injected into the prompt template for every case.",
    )
    parser.add_argument(
        "--phq9-dictionary",
        default=Path("outputs/llm_phq9_dictionary.json"),
        type=Path,
        help="JSON PHQ-9 dictionary injected into the prompt template for every case.",
    )
    parser.add_argument(
        "--cases",
        default=Path("outputs/llm_input_cases.jsonl"),
        type=Path,
        help="JSONL with one case per line, including PHQ-9 and behavioral history.",
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/llm_qualitative_predictions_raw.jsonl"),
        type=Path,
        help="Output JSONL with raw model responses and parsed qualitative analyses.",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        type=Path,
        help=(
            "Local .env file containing OPENAI_API_KEY, GEMINI_API_KEY, and/or "
            "ANTHROPIC_API_KEY, and/or credentials for the chosen Qwen backend."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai", "gemini"],
        choices=["openai", "gemini", "claude", "qwen"],
        help=(
            "Providers to run. Each provider requires its model argument. If omitted, "
            "the script can use LLM_PROVIDERS from .env."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help=(
            "LiteLLM model name for OpenAI, for example openai/gpt-4o. If omitted, "
            "the script can use OPENAI_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help=(
            "LiteLLM model name for Gemini, for example gemini/gemini-3.5-flash. "
            "If omitted, the script can use GEMINI_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--claude-model",
        default=None,
        help=(
            "LiteLLM model name for Claude, for example "
            "anthropic/claude-sonnet-5. If omitted, the script can use "
            "CLAUDE_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--qwen-model",
        default=None,
        help=(
            "LiteLLM model name for Qwen, for example ollama/qwen3.5:27b for local "
            "execution or openrouter/<MODEL_NAME> for a hosted backend. If omitted, "
            "the script can use QWEN_MODEL from .env."
        ),
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        type=float,
        help=(
            "Sampling temperature. The default is deterministic. If omitted, the "
            "script can use LLM_TEMPERATURE from .env."
        ),
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help=(
            "Optional number of cases to run for testing. If omitted, the script can "
            "use LLM_LIMIT from .env."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. If omitted, one is generated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of skipping existing case results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show planned calls without importing LiteLLM or calling APIs.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        default=2.0,
        type=float,
        help=(
            "Seconds to wait after each completed API call to reduce rate-limit errors. "
            "If omitted, the script can use LLM_REQUEST_DELAY_SECONDS from .env."
        ),
    )
    parser.add_argument(
        "--max-retries",
        default=3,
        type=int,
        help=(
            "Number of retries after the first attempt for retryable API errors. "
            "If omitted, the script can use LLM_MAX_RETRIES from .env."
        ),
    )
    parser.add_argument(
        "--retry-base-delay-seconds",
        default=15.0,
        type=float,
        help=(
            "Initial wait before retrying a retryable API error. Uses exponential "
            "backoff. If omitted, the script can use "
            "LLM_RETRY_BASE_DELAY_SECONDS from .env."
        ),
    )
    return parser.parse_args()


def flag_present(argv: list[str], flag: str) -> bool:
    return any(argument == flag or argument.startswith(f"{flag}=") for argument in argv)


def parse_provider_list(raw: str) -> list[str]:
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [provider for provider in providers if provider not in ENV_PROVIDER_NAMES]
    if invalid:
        raise ValueError(
            "Invalid provider(s) in LLM_PROVIDERS: "
            f"{invalid}. Allowed values: {sorted(ENV_PROVIDER_NAMES)}"
        )
    if not providers:
        raise ValueError("LLM_PROVIDERS is empty.")
    return providers


def env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float value for {name}: {raw!r}") from exc


def env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {name}: {raw!r}") from exc


def apply_env_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if not flag_present(argv, "--models"):
        env_providers = os.environ.get("LLM_PROVIDERS", "").strip()
        if env_providers:
            args.models = parse_provider_list(env_providers)

    if not flag_present(argv, "--openai-model"):
        args.openai_model = args.openai_model or os.environ.get("OPENAI_MODEL")

    if not flag_present(argv, "--gemini-model"):
        args.gemini_model = args.gemini_model or os.environ.get("GEMINI_MODEL")

    if not flag_present(argv, "--claude-model"):
        args.claude_model = args.claude_model or os.environ.get("CLAUDE_MODEL")

    if not flag_present(argv, "--qwen-model"):
        args.qwen_model = args.qwen_model or os.environ.get("QWEN_MODEL")

    if not flag_present(argv, "--temperature"):
        env_value = env_float("LLM_TEMPERATURE")
        if env_value is not None:
            args.temperature = env_value

    if not flag_present(argv, "--limit"):
        env_value = env_int("LLM_LIMIT")
        if env_value is not None:
            args.limit = env_value

    if not flag_present(argv, "--request-delay-seconds"):
        env_value = env_float("LLM_REQUEST_DELAY_SECONDS")
        if env_value is not None:
            args.request_delay_seconds = env_value

    if not flag_present(argv, "--max-retries"):
        env_value = env_int("LLM_MAX_RETRIES")
        if env_value is not None:
            args.max_retries = env_value

    if not flag_present(argv, "--retry-base-delay-seconds"):
        env_value = env_float("LLM_RETRY_BASE_DELAY_SECONDS")
        if env_value is not None:
            args.retry_base_delay_seconds = env_value


def resolve_env_file(path: Path) -> Path | None:
    if path.exists():
        return path

    if path == DEFAULT_ENV_FILE:
        for candidate in ENV_FILE_CANDIDATES:
            if candidate.exists():
                return candidate

    return None


def load_env_file(path: Path) -> Path | None:
    resolved_path = resolve_env_file(path)
    if resolved_path is None:
        return None

    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)

    # Some Gemini clients expect GOOGLE_API_KEY. Keep GEMINI_API_KEY as the
    # documented project variable and mirror it only if GOOGLE_API_KEY is absent.
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    if os.environ.get("QWEN_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["QWEN_API_KEY"]

    return resolved_path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def read_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL line: {exc}") from exc

            validate_case(case, line_number)
            cases.append(case)

            if limit is not None and len(cases) >= limit:
                break

    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def find_forbidden_case_keys(obj: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}"
            if key in FORBIDDEN_CASE_KEYS:
                findings.append(current_path)
            findings.extend(find_forbidden_case_keys(value, current_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            findings.extend(find_forbidden_case_keys(value, f"{path}[{index}]"))
    return findings


def validate_case(case: dict[str, Any], line_number: int) -> None:
    if list(case.keys()) != ["case_id", "phq9", "daily_history"]:
        raise ValueError(
            f"Case line {line_number}: expected case_id, phq9, and daily_history keys."
        )

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Case line {line_number}: invalid case_id.")

    phq9 = case.get("phq9")
    if not isinstance(phq9, dict):
        raise ValueError(f"{case_id}: phq9 must be an object.")

    expected_phq9_keys = [
        "item_responses",
        "item_scores",
        "total_score",
        "severity_class_5",
    ]
    if list(phq9.keys()) != expected_phq9_keys:
        raise ValueError(
            f"{case_id}: expected phq9 keys {expected_phq9_keys}, found {list(phq9.keys())}."
        )

    expected_item_keys = [f"item_{index}" for index in range(1, 10)]

    item_responses = phq9.get("item_responses")
    if not isinstance(item_responses, dict) or list(item_responses.keys()) != expected_item_keys:
        raise ValueError(f"{case_id}: invalid item_responses structure.")
    if not all(isinstance(value, str) and value.strip() for value in item_responses.values()):
        raise ValueError(f"{case_id}: item_responses must contain non-empty strings.")

    item_scores = phq9.get("item_scores")
    if not isinstance(item_scores, dict) or list(item_scores.keys()) != expected_item_keys:
        raise ValueError(f"{case_id}: invalid item_scores structure.")
    if not all(isinstance(value, int) and 0 <= value <= 3 for value in item_scores.values()):
        raise ValueError(f"{case_id}: item_scores must contain integers from 0 to 3.")

    total_score = phq9.get("total_score")
    if not isinstance(total_score, int) or not (0 <= total_score <= 27):
        raise ValueError(f"{case_id}: total_score must be an integer from 0 to 27.")
    if total_score != sum(item_scores.values()):
        raise ValueError(f"{case_id}: total_score does not match the sum of item_scores.")

    severity_class_5 = phq9.get("severity_class_5")
    if not isinstance(severity_class_5, str) or not severity_class_5.strip():
        raise ValueError(f"{case_id}: severity_class_5 must be a non-empty string.")

    daily_history = case.get("daily_history")
    if not isinstance(daily_history, list) or len(daily_history) != 14:
        raise ValueError(f"{case_id}: expected daily_history with 14 days.")

    forbidden = find_forbidden_case_keys(case)
    if forbidden:
        raise ValueError(f"{case_id}: forbidden keys found in LLM input: {forbidden}")


def build_prompt(
    prompt: str,
    feature_dictionary: dict[str, Any],
    phq9_dictionary: dict[str, Any],
    case: dict[str, Any],
) -> str:
    feature_dictionary_json = json.dumps(
        feature_dictionary,
        ensure_ascii=False,
        indent=2,
    )
    phq9_dictionary_json = json.dumps(
        phq9_dictionary,
        ensure_ascii=False,
        indent=2,
    )
    case_json = json.dumps(case, ensure_ascii=False, indent=2)
    replacements = {
        "{{FEATURE_DICTIONARY}}": feature_dictionary_json,
        "{{PHQ9_DICTIONARY}}": phq9_dictionary_json,
        "{{CASE_JSON}}": case_json,
    }

    rendered_prompt = prompt
    placeholder_count = sum(
        placeholder in prompt for placeholder in PROMPT_PLACEHOLDERS
    )

    if placeholder_count:
        for placeholder, replacement in replacements.items():
            rendered_prompt = rendered_prompt.replace(placeholder, replacement)

        missing_placeholders = [
            placeholder
            for placeholder in PROMPT_PLACEHOLDERS
            if placeholder not in prompt
        ]
        if missing_placeholders:
            missing_names = [
                PROMPT_PLACEHOLDERS[placeholder]
                for placeholder in missing_placeholders
            ]
            raise ValueError(
                "Prompt template mixes placeholder mode with missing sections. "
                f"Missing placeholders for: {missing_names}"
            )
        return rendered_prompt.rstrip() + "\n"

    return (
        prompt.rstrip()
        + "\n\nFEATURE DICTIONARY JSON:\n"
        + feature_dictionary_json
        + "\n\nPHQ-9 DICTIONARY JSON:\n"
        + phq9_dictionary_json
        + "\n\nCASE JSON:\n"
        + case_json
        + "\n"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def selected_provider_models(args: argparse.Namespace) -> list[tuple[str, str]]:
    provider_models: list[tuple[str, str]] = []

    if "openai" in args.models:
        if not args.openai_model:
            raise ValueError("Provider openai selected, but --openai-model was not set.")
        provider_models.append(("openai", args.openai_model))

    if "gemini" in args.models:
        if not args.gemini_model:
            raise ValueError("Provider gemini selected, but --gemini-model was not set.")
        provider_models.append(("gemini", args.gemini_model))

    if "claude" in args.models:
        if not args.claude_model:
            raise ValueError("Provider claude selected, but --claude-model was not set.")
        provider_models.append(("claude", args.claude_model))

    if "qwen" in args.models:
        if not args.qwen_model:
            raise ValueError("Provider qwen selected, but --qwen-model was not set.")
        provider_models.append(("qwen", args.qwen_model))

    return provider_models


def qwen_backend_requires_key(model: str) -> bool:
    local_prefixes = ("ollama/", "lm_studio/")
    return not model.startswith(local_prefixes)


def has_any_env_key(candidates: list[str]) -> bool:
    return any(os.environ.get(candidate) for candidate in candidates)


def validate_api_keys(provider_models: list[tuple[str, str]]) -> None:
    missing: list[str] = []

    if any(provider == "openai" for provider, _ in provider_models):
        if not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")

    if any(provider == "gemini" for provider, _ in provider_models):
        if not os.environ.get("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY")

    if any(provider == "claude" for provider, _ in provider_models):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")

    qwen_models = [model for provider, model in provider_models if provider == "qwen"]
    if any(qwen_backend_requires_key(model) for model in qwen_models):
        if not has_any_env_key(QWEN_KEY_CANDIDATES):
            missing.append("OPENROUTER_API_KEY or DASHSCOPE_API_KEY or QWEN_API_KEY")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing API key(s): {joined}. Add them to .env, "
            "experimento_llm\\.env, or the environment."
        )


def load_existing_result_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            provider = row.get("provider")
            model = row.get("model")
            case_id = row.get("case_id")
            status = row.get("status")
            if status != "success":
                # Only successful rows are complete. API, parse, and validation
                # errors should be retried on the next run without --overwrite.
                continue
            if isinstance(provider, str) and isinstance(model, str) and isinstance(case_id, str):
                keys.add((provider, model, case_id))
    return keys


def extract_response_text(response: Any) -> str:
    try:
        choices = response.choices
        message = choices[0].message
        content = message.content
    except (AttributeError, IndexError):
        choices = response["choices"]
        message = choices[0]["message"]
        content = message["content"]

    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def parse_model_json(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = response_text.strip()
    candidates = [stripped]

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed, None
        return None, "Parsed JSON is not an object."

    return None, "Response is not valid JSON."


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    sentence_like_chunks = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", stripped)
        if chunk.strip()
    ]
    if sentence_like_chunks:
        return len(sentence_like_chunks)

    return 1


def validate_parsed_response(
    parsed_response: dict[str, Any],
    expected_case_id: str,
) -> list[str]:
    errors: list[str] = []

    expected_keys = {
        "case_id",
        "behavioral_summary",
        "phq9_summary",
        "cross_analysis",
        "unsupported_or_weak_links",
        "uncertainties",
        "overall_assessment",
    }
    actual_keys = set(parsed_response.keys())
    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        errors.append(
            "unexpected response keys: "
            f"missing={missing_keys}, extra={extra_keys}"
        )

    case_id = parsed_response.get("case_id")
    if case_id != expected_case_id:
        errors.append(
            f"case_id mismatch: expected {expected_case_id}, found {case_id!r}"
        )

    behavioral_summary = parsed_response.get("behavioral_summary")
    if not isinstance(behavioral_summary, str) or not behavioral_summary.strip():
        errors.append("behavioral_summary is missing or empty")
    elif not (1 <= count_sentences(behavioral_summary) <= 3):
        errors.append("behavioral_summary must contain 1 to 3 sentences")

    phq9_summary = parsed_response.get("phq9_summary")
    if not isinstance(phq9_summary, str) or not phq9_summary.strip():
        errors.append("phq9_summary is missing or empty")
    elif not (1 <= count_sentences(phq9_summary) <= 3):
        errors.append("phq9_summary must contain 1 to 3 sentences")

    cross_analysis = parsed_response.get("cross_analysis")
    if not isinstance(cross_analysis, list):
        errors.append("cross_analysis must be a list")
    else:
        expected_cross_keys = {
            "phq9_item",
            "symptom_label",
            "support_level",
            "behavioral_evidence",
            "caution",
        }
        for index, entry in enumerate(cross_analysis):
            if not isinstance(entry, dict):
                errors.append(f"cross_analysis[{index}] must be an object")
                continue
            actual_cross_keys = set(entry.keys())
            if actual_cross_keys != expected_cross_keys:
                missing_cross_keys = sorted(expected_cross_keys - actual_cross_keys)
                extra_cross_keys = sorted(actual_cross_keys - expected_cross_keys)
                errors.append(
                    "cross_analysis"
                    f"[{index}] unexpected keys: missing={missing_cross_keys}, "
                    f"extra={extra_cross_keys}"
                )
                continue

            phq9_item = entry.get("phq9_item")
            if phq9_item not in {f"item_{item_index}" for item_index in range(1, 10)}:
                errors.append(f"cross_analysis[{index}].phq9_item is invalid: {phq9_item!r}")

            symptom_label = entry.get("symptom_label")
            if not isinstance(symptom_label, str) or not symptom_label.strip():
                errors.append(f"cross_analysis[{index}].symptom_label is missing or empty")

            support_level = entry.get("support_level")
            if support_level not in ALLOWED_SUPPORT_LEVELS:
                errors.append(
                    f"cross_analysis[{index}].support_level is invalid: {support_level!r}"
                )

            behavioral_evidence = entry.get("behavioral_evidence")
            if not isinstance(behavioral_evidence, str) or not behavioral_evidence.strip():
                errors.append(
                    f"cross_analysis[{index}].behavioral_evidence is missing or empty"
                )
            elif not (1 <= count_sentences(behavioral_evidence) <= 2):
                errors.append(
                    "cross_analysis"
                    f"[{index}].behavioral_evidence must contain 1 or 2 sentences"
                )

            caution = entry.get("caution")
            if not isinstance(caution, str) or not caution.strip():
                errors.append(f"cross_analysis[{index}].caution is missing or empty")
            elif not (1 <= count_sentences(caution) <= 2):
                errors.append(
                    f"cross_analysis[{index}].caution must contain 1 or 2 sentences"
                )

    unsupported_or_weak_links = parsed_response.get("unsupported_or_weak_links")
    if not isinstance(unsupported_or_weak_links, list):
        errors.append("unsupported_or_weak_links must be a list")
    elif not all(isinstance(item, str) and item.strip() for item in unsupported_or_weak_links):
        errors.append("unsupported_or_weak_links must contain only non-empty strings")

    uncertainties = parsed_response.get("uncertainties")
    if not isinstance(uncertainties, list):
        errors.append("uncertainties must be a list")
    elif not all(isinstance(item, str) and item.strip() for item in uncertainties):
        errors.append("uncertainties must contain only non-empty strings")

    overall_assessment = parsed_response.get("overall_assessment")
    if not isinstance(overall_assessment, str) or not overall_assessment.strip():
        errors.append("overall_assessment is missing or empty")

    return errors


def make_result_row(
    run_id: str,
    provider: str,
    model: str,
    case_id: str,
    status: str,
    prompt_hash: str,
    temperature: float,
    response_text: str | None,
    parsed_response: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "provider": provider,
        "model": model,
        "case_id": case_id,
        "status": status,
        "prompt_hash": prompt_hash,
        "temperature": temperature,
        "response_text": response_text,
        "parsed_response": parsed_response,
        "error": error,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def load_completion_function() -> Any:
    try:
        from litellm import completion
        
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: litellm. Install it with: pip install litellm"
        ) from exc
    return completion


def run_completion(
    completion_function: Any,
    model: str,
    prompt: str,
    temperature: float,
) -> str:
    # A new messages list is created for each call to avoid carrying context
    # from one participant case into another.
    messages = [{"role": "user", "content": prompt}]
    response = completion_function(model=model, messages=messages, temperature=temperature)
    return extract_response_text(response)


def is_retryable_api_error(error: Exception) -> bool:
    text = str(error).lower()
    retryable_terms = [
        "too many requests",
        "rate limit",
        "ratelimit",
        "429",
        "resource exhausted",
        "quota",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "service unavailable",
        "503",
    ]
    return any(term in text for term in retryable_terms)


def run_completion_with_retries(
    completion_function: Any,
    provider: str,
    model: str,
    case_id: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    retry_base_delay_seconds: float,
) -> str:
    attempt = 0
    while True:
        try:
            return run_completion(completion_function, model, prompt, temperature)
        except Exception as exc:
            if not is_retryable_api_error(exc) or attempt >= max_retries:
                raise

            attempt += 1
            wait_seconds = retry_base_delay_seconds * (2 ** (attempt - 1))
            print(
                f"RETRY {provider} {model} {case_id}: "
                f"attempt {attempt}/{max_retries} after {wait_seconds:.1f}s "
                f"because of API rate/temporary error"
            )
            time.sleep(wait_seconds)


def run_experiment(args: argparse.Namespace) -> None:
    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds must be zero or positive.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be zero or positive.")
    if args.retry_base_delay_seconds < 0:
        raise ValueError("--retry-base-delay-seconds must be zero or positive.")

    loaded_env_file = load_env_file(args.env_file)
    apply_env_defaults(args, sys.argv[1:])
    provider_models = selected_provider_models(args)
    prompt_template = read_text(args.prompt)
    feature_dictionary = read_json(args.feature_dictionary)
    phq9_dictionary = read_json(args.phq9_dictionary)
    cases = read_cases(args.cases, args.limit)
    run_id = args.run_id or default_run_id()

    planned_calls = [
        (provider, model, case)
        for provider, model in provider_models
        for case in cases
    ]

    if args.dry_run:
        first_prompt = build_prompt(
            prompt_template,
            feature_dictionary,
            phq9_dictionary,
            cases[0],
        )
        print("Dry run: no API calls were made and no output file was written.")
        print(f"Loaded env file: {loaded_env_file or 'not found'}")
        print(f"Run ID: {run_id}")
        print(f"Feature dictionary: {args.feature_dictionary}")
        print(f"PHQ-9 dictionary: {args.phq9_dictionary}")
        print(f"Cases loaded: {len(cases)}")
        print(f"Planned calls: {len(planned_calls)}")
        print(f"Providers/models: {provider_models}")
        print(f"Temperature: {args.temperature}")
        print(f"Request delay seconds: {args.request_delay_seconds}")
        print(f"Max retries: {args.max_retries}")
        print(f"Retry base delay seconds: {args.retry_base_delay_seconds}")
        print(f"First case ID: {cases[0]['case_id']}")
        print(f"First prompt SHA256: {sha256_text(first_prompt)}")
        print("Validation: PASS")
        return

    validate_api_keys(provider_models)
    completion_function = load_completion_function()

    if args.overwrite:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        existing_keys: set[tuple[str, str, str]] = set()
    else:
        existing_keys = load_existing_result_keys(args.output)

    completed = 0
    skipped = 0
    failed = 0

    for call_index, (provider, model, case) in enumerate(planned_calls, start=1):
        case_id = str(case["case_id"])
        result_key = (provider, model, case_id)
        if result_key in existing_keys:
            skipped += 1
            print(f"SKIP {provider} {model} {case_id}: existing result")
            continue

        full_prompt = build_prompt(
            prompt_template,
            feature_dictionary,
            phq9_dictionary,
            case,
        )
        prompt_hash = sha256_text(full_prompt)

        try:
            response_text = run_completion_with_retries(
                completion_function,
                provider,
                model,
                case_id,
                full_prompt,
                args.temperature,
                args.max_retries,
                args.retry_base_delay_seconds,
            )
            parsed_response, parse_error = parse_model_json(response_text)

            if parse_error:
                row = make_result_row(
                    run_id,
                    provider,
                    model,
                    case_id,
                    "parse_error",
                    prompt_hash,
                    args.temperature,
                    response_text,
                    None,
                    parse_error,
                )
                failed += 1
            else:
                assert parsed_response is not None
                validation_errors = validate_parsed_response(parsed_response, case_id)
                if validation_errors:
                    row = make_result_row(
                        run_id,
                        provider,
                        model,
                        case_id,
                        "validation_error",
                        prompt_hash,
                        args.temperature,
                        response_text,
                        parsed_response,
                        "; ".join(validation_errors),
                    )
                    failed += 1
                else:
                    row = make_result_row(
                        run_id,
                        provider,
                        model,
                        case_id,
                        "success",
                        prompt_hash,
                        args.temperature,
                        response_text,
                        parsed_response,
                        None,
                    )
                    completed += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            row = make_result_row(
                run_id,
                provider,
                model,
                case_id,
                "api_error",
                prompt_hash,
                args.temperature,
                None,
                None,
                str(exc),
            )
            failed += 1

        append_jsonl(args.output, row)
        existing_keys.add(result_key)
        print(f"{row['status'].upper()} {provider} {model} {case_id}")

        if args.request_delay_seconds > 0 and call_index < len(planned_calls):
            time.sleep(args.request_delay_seconds)

    print(f"Output: {args.output}")
    print(f"Run ID: {run_id}")
    print(f"Completed successful calls: {completed}")
    print(f"Skipped existing calls: {skipped}")
    print(f"Calls with errors: {failed}")


def main() -> None:
    try:
        args = parse_args()
        run_experiment(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
