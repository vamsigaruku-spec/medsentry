# ================================================================
# MEDSENTRY GROQ MODEL
# ================================================================

import os

from groq import Groq


# ================================================================
# CONFIGURATION
# ================================================================

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# ================================================================
# API KEY
# ================================================================

def _get_api_key():

    # ------------------------------------------------------------
    # Environment variable
    # ------------------------------------------------------------

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if api_key:
        return api_key

    # ------------------------------------------------------------
    # Streamlit Cloud Secrets
    # ------------------------------------------------------------

    try:

        import streamlit as st

        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if api_key:
            return api_key

    except Exception:
        pass

    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Add GROQ_API_KEY to Streamlit Secrets."
    )


# ================================================================
# GROQ CLIENT
# ================================================================

def _get_client():

    api_key = _get_api_key()

    return Groq(
        api_key=api_key
    )


# ================================================================
# GENERATE ANSWER
# ================================================================

def generate_answer(prompt):

    if not prompt or not str(prompt).strip():

        raise ValueError(
            "Prompt cannot be empty."
        )

    client = _get_client()

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are MedSentry, an "
                    "evidence-grounded medical "
                    "information assistant. "
                    "Follow the instructions contained "
                    "in the user prompt."
                )
            },
            {
                "role": "user",
                "content": str(prompt)
            }
        ],

        temperature=0.1,

        max_tokens=800,

        stream=False
    )

    # ------------------------------------------------------------
    # Validate response
    # ------------------------------------------------------------

    if response is None:

        raise RuntimeError(
            "Groq returned no response."
        )

    if not response.choices:

        raise RuntimeError(
            "Groq returned no choices."
        )

    message = response.choices[0].message

    if message is None:

        raise RuntimeError(
            "Groq returned an empty message."
        )

    answer = message.content

    if answer is None:

        raise RuntimeError(
            "Groq returned empty content."
        )

    answer = str(answer).strip()

    if not answer:

        raise RuntimeError(
            "Groq returned an empty answer."
        )

    return answer


# ================================================================
# MODEL HEALTH CHECK
# ================================================================

def test_model():

    test_prompt = """
QUESTION:
What is hypertension?

RETRIEVED EVIDENCE:
Hypertension is high blood pressure.

Return exactly:

{
  "answer": "A short evidence-grounded answer.",
  "safety": "A short appropriate safety note."
}
"""

    return generate_answer(
        test_prompt
    )


# ================================================================
# DIRECT TEST
# ================================================================

if __name__ == "__main__":

    try:

        result = test_model()

        print("=" * 60)
        print("MEDSENTRY GROQ MODEL TEST")
        print("=" * 60)

        print(result)

        print("=" * 60)

    except Exception as e:

        print("=" * 60)
        print("MEDSENTRY GROQ MODEL ERROR")
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 60)
