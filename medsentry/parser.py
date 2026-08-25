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
    Validated structured response returned by MedSentry.
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
# JSON PARSER
# ================================================================

def _parse_json(text: str) -> Optional[MedSentryResponse]:

    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            return None

        return MedSentryResponse.model_validate(data)

    except (
        json.JSONDecodeError,
        ValidationError,
        TypeError,
        ValueError
    ):
        return None


# ================================================================
# EXTRACT JSON OBJECT
# ================================================================

def _extract_json_object(text: str) -> Optional[str]:

    # Markdown JSON block
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    # Find first JSON object in plain text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:

        candidate = text[start:end + 1].strip()

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


# ================================================================
# ANSWER / SAFETY FORMAT PARSER
# ================================================================

def _parse_answer_safety_format(
    text: str
) -> MedSentryResponse:

    answer_match = re.search(
        r"ANSWER\s*:\s*(.*?)(?=\n\s*SAFETY\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = text.strip()

    safety_match = re.search(
        r"SAFETY\s*:\s*(.*)$",
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

def parse_model_output(raw_response):

    """
    Parse Groq output into a validated MedSentry response.

    Supported formats:

    1. JSON
    2. Markdown JSON
    3. ANSWER / SAFETY format
    4. Plain text fallback
    """

    if raw_response is None:

        result = MedSentryResponse(
            answer="The model did not provide a usable answer.",
            safety=""
        )

        return {
            **result.model_dump(),
            "raw_response": ""
        }

    text = str(raw_response).strip()

    if not text:

        result = MedSentryResponse(
            answer="The model did not provide a usable answer.",
            safety=""
        )

        return {
            **result.model_dump(),
            "raw_response": ""
        }

    # ------------------------------------------------------------
    # Attempt 1: Direct JSON
    # ------------------------------------------------------------

    parsed = _parse_json(text)

    if parsed is not None:

        return {
            **parsed.model_dump(),
            "raw_response": text
        }

    # ------------------------------------------------------------
    # Attempt 2: JSON embedded in response
    # ------------------------------------------------------------

    json_object = _extract_json_object(text)

    if json_object:

        parsed = _parse_json(json_object)

        if parsed is not None:

            return {
                **parsed.model_dump(),
                "raw_response": text
            }

    # ------------------------------------------------------------
    # Attempt 3: ANSWER / SAFETY format
    # ------------------------------------------------------------

    parsed = _parse_answer_safety_format(text)

    return {
        **parsed.model_dump(),
        "raw_response": text
    }


# ================================================================
# VALIDATION HELPER
# ================================================================

def validate_output(data):

    return MedSentryResponse.model_validate(data)


# ================================================================
# JSON EXPORT HELPER
# ================================================================

def response_to_json(response):

    if isinstance(response, dict):

        response = MedSentryResponse.model_validate(response)

    return response.model_dump_json(
        indent=2
    )
