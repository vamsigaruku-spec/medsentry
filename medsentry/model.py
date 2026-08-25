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
# GET GROQ API KEY
# ================================================================

def _get_api_key():
    """
    Get the Groq API key securely.

    Priority:
    1. Environment variable
    2. Streamlit secrets
    """

    # Environment variable
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key

    # Streamlit Cloud secrets
    try:

        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Add GROQ_API_KEY to Streamlit Secrets."
    )


# ================================================================
# CREATE GROQ CLIENT
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
    """
    Generate an answer using Groq.

    The prompt is created by prompt.py and contains:
    - System instructions
    - User question
    - Retrieved RAG evidence
    - Medical safety rules
    """

    if not prompt or not str(prompt).strip():

        raise ValueError(
            "Prompt cannot be empty."
        )

    client = _get_client()

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": str(prompt)
            }
        ],

        temperature=0.2,

        max_tokens=800,

        stream=False
    )

    # ------------------------------------------------------------
    # Extract generated text
    # ------------------------------------------------------------

    if not response or not response.choices:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    answer = response.choices[0].message.content

    if not answer:

        raise RuntimeError(
            "Groq did not return any text."
        )

    return answer.strip()


# ================================================================
# SIMPLE MODEL TEST
# ================================================================

def test_model():

    test_prompt = """
You are MedSentry, an evidence-grounded medical information assistant.

Answer only from the provided evidence.

QUESTION:
What is hypertension?

EVIDENCE:
Hypertension is high blood pressure.

Return:

ANSWER:
<answer>

SAFETY:
<brief safety note>
"""

    return generate_answer(test_prompt)


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

        print(str(e))

        print("=" * 60)
