"""LLM access helpers (secure key handling).

The OpenAI key is read SERVER-SIDE only, from either an environment variable or Streamlit
secrets — never hardcoded, never returned to the browser, never logged. `get_openai_key`
returns None if no key is configured, so callers can fall back to the rule-based path.
"""
from __future__ import annotations

import os


def get_openai_key() -> str | None:
    """Return the OpenAI API key from env or Streamlit secrets, or None if unset."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("OPENAI_API_KEY")   # server-side only
    except Exception:
        return None


def have_openai_key() -> bool:
    return bool(get_openai_key())


DEFAULT_MODEL = "gpt-4o-mini"


def chat_text(messages: list[dict], system: str, model: str = DEFAULT_MODEL,
              temperature: float = 0.2, json_mode: bool = False) -> str:
    """One chat completion (raises if no key). All OpenAI calls funnel through here."""
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_key())
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system}, *messages], **kwargs)
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict], system: str, model: str = DEFAULT_MODEL,
              temperature: float = 0.2) -> dict:
    import json
    return json.loads(chat_text(messages, system, model, temperature, json_mode=True))
