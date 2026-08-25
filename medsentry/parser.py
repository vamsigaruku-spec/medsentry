# ================================================================
# MEDSENTRY OUTPUT PARSER
# ================================================================

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


# ================================================================
# OUTPUT SCHEMA
# ================================================================

class MedSentryResponse(BaseModel):
    """
    Structured output schema for MedSentry.
    """

    answer: str = Field(
        ...,
        description="Evidence-grounded medical answer."
    )

    safety: str = Field(
        default="",
        description="Medical safety note when relevant."
    )


# ================================================================
# PARSE JSON RESPONSE
# ================================================================

def _parse_json(text: str) -> Optional[MedSentryResponse]:
    """
    Try to parse a JSON response into the Pydantic schema.
    """

    try:

        data = json.loads(text)

        return MedSentryResponse.model_validate(data)

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError
    ):

        return None


# ================================================================
# EXTRACT JSON FROM MARKDOWN
# ================================================================

def _extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON from a markdown code block.
    """

    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:

        return match.group(1).strip()

    return None


# ================================================================
# LEGACY ANSWER / SAFETY PARSER
# ================================================================

def _parse_answer_safety_format(
    text: str
) -> MedSentryResponse:
    """
    Fallback parser for:

    ANSWER:
    ...

    SAFETY:
    ...
    """

    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=\n\s*SAFETY:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if answer_match:

        answer = answer_match.group(1).strip()

    else:

        answer = text.strip()


    safety_match = re.search(
        r"SAFETY:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if safety_match:

        safety = safety_match.group(1).strip()

    else:

        safety = ""


    if not answer:

        answer = (
            "The model did not provide a usable answer."
        )


    return MedSentryResponse(
        answer=answer,
        safety=safety
    )


# ================================================================
# MAIN PARSER
# ================================================================

def parse_model_output(
    raw_response
):
    """
    Parse and validate the model response.

    Supported formats:

    1. JSON

    {
        "answer": "...",
        "safety": "..."
    }

    2. Markdown JSON block

    ```json
    {
        "answer": "...",
        "safety": "..."
    }
    ```

    3. Legacy format

    ANSWER:
    ...

    SAFETY:
    ...

    Returns a structured dictionary.
    """

    # ------------------------------------------------------------
    # Empty response
    # ------------------------------------------------------------

    if not raw_response:

        result = MedSentryResponse(
            answer=(
                "The model did not provide a usable answer."
            ),
            safety=""
        )

        return {
            **result.model_dump(),
            "raw_response": ""
        }


    text = str(
        raw_response
    ).strip()


    # ------------------------------------------------------------
    # Attempt 1: Direct JSON
    # ------------------------------------------------------------

    parsed = _parse_json(
        text
    )

    if parsed is not None:

        return {
            **parsed.model_dump(),
            "raw_response": text
        }


    # ------------------------------------------------------------
    # Attempt 2: JSON inside markdown
    # ------------------------------------------------------------

    json_block = _extract_json_block(
        text
    )

    if json_block:

        parsed = _parse_json(
            json_block
        )

        if parsed is not None:

            return {
                **parsed.model_dump(),
                "raw_response": text
            }


    # ------------------------------------------------------------
    # Attempt 3: ANSWER / SAFETY format
    # ------------------------------------------------------------

    parsed = _parse_answer_safety_format(
        text
    )


    return {
        **parsed.model_dump(),
        "raw_response": text
    }


# ================================================================
# VALIDATION HELPER
# ================================================================

def validate_output(
    data
):
    """
    Validate an already structured response.

    Returns:

        MedSentryResponse

    or raises:

        pydantic.ValidationError
    """

    return MedSentryResponse.model_validate(
        data
    )


# ================================================================
# JSON EXPORT HELPER
# ================================================================

def response_to_json(
    response: MedSentryResponse
):
    """
    Convert the validated Pydantic response
    into JSON.
    """

    return response.model_dump_json(
        indent=2
    )
