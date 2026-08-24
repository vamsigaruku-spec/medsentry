"""
MedSentry LLM Model

Handles the language-model call using Google Gemini.
The API key is loaded from Streamlit secrets.
"""

import streamlit as st
from google import genai


def get_client():
    """
    Create Gemini client using Streamlit secrets.
    """

    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


def generate_answer(prompt: str) -> str:
    """
    Send the prepared prompt to Gemini and return
    the raw model response.
    """

    client = get_client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if not response or not response.text:
        raise RuntimeError(
            "The language model returned an empty response."
        )

    return response.text.strip()
