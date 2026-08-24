"""
MedSentry Output Parser

Validates and normalizes the structured response
returned by the language model.
"""

import json


def parse_model_output(raw_output: str) -> dict:
    """
    Parse the model response into a validated dictionary.
    """

    if not raw_output:
        raise ValueError("Empty model response.")

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model response is not valid JSON."
        ) from exc

    required_fields = [
        "answer",
        "grounded",
        "requires_clinician_review",
        "safety_note"
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Missing required output field: {field}"
            )

    if not isinstance(data["answer"], str):
        raise ValueError("'answer' must be a string.")

    if not isinstance(data["grounded"], bool):
        raise ValueError("'grounded' must be boolean.")

    if not isinstance(
        data["requires_clinician_review"],
        bool
    ):
        raise ValueError(
            "'requires_clinician_review' must be boolean."
        )

    if not isinstance(data["safety_note"], str):
        raise ValueError("'safety_note' must be a string.")

    return data
