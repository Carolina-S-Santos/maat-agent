from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ITEM_KEYS = [f"item_{index}" for index in range(1, 10)]

ITEM_LABELS = {
    "item_1": "Little interest or pleasure in doing things",
    "item_2": "Feeling down, depressed, or hopeless",
    "item_3": "Trouble falling or staying asleep, or sleeping too much",
    "item_4": "Feeling tired or having little energy",
    "item_5": "Poor appetite or overeating",
    "item_6": (
        "Feeling bad about yourself or that you are a failure or have let yourself "
        "or your family down"
    ),
    "item_7": (
        "Trouble concentrating on things, such as reading the newspaper or "
        "watching television"
    ),
    "item_8": (
        "Moving or speaking so slowly that other people could have noticed, or "
        "the opposite: being fidgety or restless"
    ),
    "item_9": "Thoughts that you would be better off dead or of hurting yourself",
}

RESPONSE_SCALE = {
    "0": "Not at all",
    "1": "Several days",
    "2": "More than half the days",
    "3": "Nearly every day",
}

SEVERITY_CLASS_5_CATEGORIES = {
    "0-4": "minimal",
    "5-9": "mild",
    "10-14": "moderate",
    "15-19": "moderately_severe",
    "20-27": "severe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the PHQ-9 dictionary used in LLM prompts."
    )
    parser.add_argument(
        "--output",
        default=Path("outputs/llm_phq9_dictionary.json"),
        type=Path,
        help="Output JSON PHQ-9 dictionary.",
    )
    return parser.parse_args()


def build_dictionary() -> dict[str, Any]:
    return {
        "questionnaire": {
            "name": "PHQ-9",
            "description": "Questionnaire about depressive symptoms over the last two weeks.",
        },
        "response_scale": RESPONSE_SCALE,
        "items": {
            item_key: {"label": ITEM_LABELS[item_key]}
            for item_key in ITEM_KEYS
        },
        "derived_fields": {
            "total_score": {
                "description": "Sum of the 9 item scores.",
                "range": [0, 27],
            },
            "severity_class_5": {
                "description": (
                    "Five-level severity category derived from the total PHQ-9 score."
                ),
                "categories": SEVERITY_CLASS_5_CATEGORIES,
            },
        },
    }


def validate_dictionary(dictionary: dict[str, Any]) -> None:
    questionnaire = dictionary.get("questionnaire")
    if not isinstance(questionnaire, dict):
        raise ValueError("Dictionary must contain a 'questionnaire' object.")

    if questionnaire.get("name") != "PHQ-9":
        raise ValueError("Questionnaire name must be 'PHQ-9'.")

    response_scale = dictionary.get("response_scale")
    if response_scale != RESPONSE_SCALE:
        raise ValueError("Unexpected PHQ-9 response scale definition.")

    items = dictionary.get("items")
    if not isinstance(items, dict):
        raise ValueError("Dictionary must contain an 'items' object.")

    if list(items.keys()) != ITEM_KEYS:
        raise ValueError(f"Items must be ordered as: {ITEM_KEYS}")

    for item_key in ITEM_KEYS:
        item = items[item_key]
        if not isinstance(item, dict):
            raise ValueError(f"{item_key} must be an object.")
        if list(item.keys()) != ["label"]:
            raise ValueError(f"{item_key} must contain only the 'label' field.")
        if item["label"] != ITEM_LABELS[item_key]:
            raise ValueError(f"Unexpected label for {item_key}.")

    derived_fields = dictionary.get("derived_fields")
    if not isinstance(derived_fields, dict):
        raise ValueError("Dictionary must contain a 'derived_fields' object.")

    total_score = derived_fields.get("total_score")
    if not isinstance(total_score, dict):
        raise ValueError("Missing 'derived_fields.total_score'.")
    if total_score.get("range") != [0, 27]:
        raise ValueError("Unexpected total_score range.")

    severity_class_5 = derived_fields.get("severity_class_5")
    if not isinstance(severity_class_5, dict):
        raise ValueError("Missing 'derived_fields.severity_class_5'.")
    if severity_class_5.get("categories") != SEVERITY_CLASS_5_CATEGORIES:
        raise ValueError("Unexpected severity_class_5 categories.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dictionary = build_dictionary()
    validate_dictionary(dictionary)
    write_json(args.output, dictionary)

    print(f"Saved PHQ-9 dictionary: {args.output}")
    print(f"Items: {len(ITEM_KEYS)}")
    print(f"Response scale values: {len(RESPONSE_SCALE)}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
