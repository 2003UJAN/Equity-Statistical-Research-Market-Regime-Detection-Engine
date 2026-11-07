
import streamlit as st
from google import genai
from google.genai import types
import json


def get_gemini_client():
    """Initialize Gemini API client."""

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(api_key=api_key)

    except Exception:
        return None


def explain_quantitative_results(
    research_context,
    question,
    conversation_history=None
):
    """Explain Python-calculated financial metrics using Gemini."""

    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API is not configured. "
            "Please check your secrets.toml file."
        )

    model_name = st.secrets.get(
        "GEMINI_MODEL",
        "gemini-2.5-flash"
    )

    system_instruction = """
    You are a professional Quantitative Equity Research Assistant.

    Your responsibilities:

    1. Explain financial statistics.
    2. Interpret stock returns and volatility.
    3. Explain correlation and covariance.
    4. Interpret CAPM, alpha and beta.
    5. Explain Sharpe Ratio, VaR and CVaR.
    6. Interpret ADF stationarity tests.
    7. Explain GARCH volatility models.
    8. Interpret K-Means market regimes.
    9. Generate institutional equity research summaries.

    RULES:

    - Never invent financial metrics.
    - Use only supplied numerical results.
    - Do not modify Python-calculated values.
    - Explain the financial significance of results.
    - Identify limitations and assumptions.
    - Distinguish correlation from causation.
    - Do not guarantee future returns.
    - Do not provide personalized investment advice.
    - Use professional research language.
    - Use headings and bullet points.
    """

    history = ""

    if conversation_history:
        for message in conversation_history[-6:]:
            history += (
                f"\n{message['role']}: "
                f"{message['content']}"
            )

    context = json.dumps(
        research_context,
        indent=2,
        default=str
    )

    prompt = f"""
    QUANTITATIVE RESEARCH DATA:

    {context}

    PREVIOUS CONVERSATION:

    {history}

    USER QUESTION:

    {question}

    Provide a structured quantitative research explanation.

    Include:
    1. Definition of relevant metrics.
    2. Interpretation of calculated results.
    3. Financial implications.
    4. Risk considerations.
    5. Research limitations.

    Do not invent unavailable data.
    """

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=1500
            )
        )

        if response.text:
            return response.text

        return "Gemini did not return an explanation."

    except Exception as error:

        return (
            "Gemini API error. Please verify your API key, "
            "model availability and API quota. "
            f"Details: {error}"
        )
