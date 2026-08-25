# ================================================================
# MEDSENTRY OUTPUT PARSER
# ================================================================

import re


def parse_model_output(raw_response):
    """
    Parse the Gemini response into a structured dictionary.

    Expected format:

    ANSWER:
    <medical answer>

    SAFETY:
    <safety note>
    """

    if not raw_response:
        return {
            "answer": "",
            "safety": "",
            "raw_response": ""
        }

    text = str(raw_response).strip()

    # ------------------------------------------------------------
    # Extract ANSWER section
    # ------------------------------------------------------------

    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?=\n\s*SAFETY:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = text

    # ------------------------------------------------------------
    # Extract SAFETY section
    # ------------------------------------------------------------

    safety_match = re.search(
        r"SAFETY:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if safety_match:
        safety = safety_match.group(1).strip()
    else:
        safety = ""

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------

    if not answer:
        answer = (
            "The model did not provide a usable answer."
        )

    return {
        "answer": answer,
        "safety": safety,
        "raw_response": text
    }
