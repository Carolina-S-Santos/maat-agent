from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_TEST_MESSAGE = (
    'Return only this JSON object: {"connection_test":"ok"}'
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a minimal test prompt to selected providers via LiteLLM."
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
        help="Providers to test. If omitted, the script can use LLM_PROVIDERS from .env.",
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
        "--message",
        default=DEFAULT_TEST_MESSAGE,
        help="Minimal message sent to each selected model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without importing LiteLLM or calling APIs.",
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

    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    if os.environ.get("QWEN_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["QWEN_API_KEY"]

    return resolved_path


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


def load_completion_function() -> Any:
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: litellm. Install it with: "
            "pip install -r experimento_llm\\requirements.txt"
        ) from exc
    return completion


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
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


def serialize_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        try:
            return response.model_dump_json(indent=2)
        except Exception:
            pass

    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            pass

    if hasattr(response, "dict"):
        try:
            return json.dumps(response.dict(), ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        return json.dumps(response, ensure_ascii=False, indent=2)
    except TypeError:
        return repr(response)


def test_model(completion_function: Any, provider: str, model: str, message: str) -> bool:
    messages = [{"role": "user", "content": message}]
    try:
        response = completion_function(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=40,
        )
        response_text = extract_response_text(response)
    except Exception as exc:
        print(f"FAIL {provider} {model}: {exc}")
        return False

    print(f"OK {provider} {model}: {response_text}")
    return True


def debug_model(completion_function: Any, provider: str, model: str, message: str) -> bool:
    messages = [{"role": "user", "content": message}]
    try:
        response = completion_function(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=80,
        )
    except Exception as exc:
        print(f"FAIL {provider} {model}: {exc}")
        return False

    print(f"RAW {provider} {model}:")
    print(serialize_response(response))
    return True


def main() -> None:
    try:
        args = parse_args()
        loaded_env_file = load_env_file(args.env_file)
        apply_env_defaults(args, sys.argv[1:])
        provider_models = selected_provider_models(args)

        if args.dry_run:
            print("Dry run: no API calls were made.")
            print(f"Loaded env file: {loaded_env_file or 'not found'}")
            print(f"Providers/models: {provider_models}")
            print(f"Message: {args.message}")
            print("Validation: PASS")
            return

        validate_api_keys(provider_models)
        completion_function = load_completion_function()

        success_count = 0
        for provider, model in provider_models:
            if test_model(completion_function, provider, model, args.message):
                success_count += 1

        total = len(provider_models)
        print(f"Connection test summary: {success_count}/{total} succeeded.")
        if success_count != total:
            raise SystemExit(1)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
