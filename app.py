"""
app.py

Weather AI Agent -- a Streamlit chat app powered by Groq's function calling
(OpenAI-compatible API) and the free, key-less Open-Meteo APIs for real
weather data.

Dependencies (see requirements.txt):
    streamlit
    openai
    requests
    python-dotenv

Get a free Groq API key at: https://console.groq.com/keys
Set it as GROQ_API_KEY in a local .env file to avoid asking every visitor for
their own key; the sidebar field remains as an optional override.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from weather_tool import get_current_weather

load_dotenv()
DEFAULT_API_KEY = os.environ.get("GROQ_API_KEY", "")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
# Caps how many tool-call round trips the model may make for a single user
# message, so one ambiguous location can't spiral into repeated API calls.
MAX_TOOL_CALLS_PER_MESSAGE = 2

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable weather assistant. Use the "
    "get_current_weather tool whenever the user asks about weather, "
    "temperature, or conditions in a specific place. Answer follow-up "
    "questions (e.g. what to wear, whether to bring an umbrella) using the "
    "most recent weather data already retrieved in this conversation "
    "whenever possible, only calling the tool again if the user asks about "
    "a new location or wants a fresh reading. If the tool returns an error "
    "that the location could not be found, do NOT retry it with a guessed "
    "variation of the spelling -- immediately tell the user the location "
    "wasn't found and ask them to clarify or try a nearby major city."
)

WEATHER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current real-time weather (temperature, humidity, "
            "precipitation, wind speed, and general conditions) for a given "
            "city."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "The city name to get weather for, e.g. 'Paris', "
                        "'Tokyo', or 'San Francisco, CA'."
                    ),
                }
            },
            "required": ["location"],
        },
    },
}


def init_session_state() -> None:
    """Initialize Streamlit session state on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def get_client(api_key: str) -> OpenAI:
    """Return an OpenAI-compatible client pointed at Groq's API."""
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def run_tool_call(tool_call: Any) -> dict[str, Any]:
    """Execute a single tool call requested by the model and return its message."""
    function_name = tool_call.function.name

    if function_name != "get_current_weather":
        result: dict[str, Any] = {"error": f"Unknown tool requested: {function_name}"}
    else:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}
        try:
            result = get_current_weather(**arguments)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
            result = {"error": f"Tool '{function_name}' failed: {exc}"}

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": json.dumps(result),
    }


def call_with_retry(client: OpenAI, messages: list[dict[str, Any]]) -> Any:
    """Call the model, retrying on transient rate-limit (429) and server (5xx) errors."""
    last_error: Optional[APIError] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=[WEATHER_TOOL_SCHEMA],
                tool_choice="auto",
            )
        except (RateLimitError, InternalServerError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error  # type: ignore[misc]


def get_assistant_response(client: OpenAI) -> str:
    """Call the model, resolve any tool calls (capped), and return the final reply text."""
    response = call_with_retry(client, st.session_state.messages)
    response_message = response.choices[0].message
    st.session_state.messages.append(response_message.model_dump(exclude_none=True))

    tool_call_rounds = 0
    while response_message.tool_calls and tool_call_rounds < MAX_TOOL_CALLS_PER_MESSAGE:
        tool_call_rounds += 1
        with st.spinner("Fetching live weather data..."):
            for tool_call in response_message.tool_calls:
                st.session_state.messages.append(run_tool_call(tool_call))

        response = call_with_retry(client, st.session_state.messages)
        response_message = response.choices[0].message
        st.session_state.messages.append(response_message.model_dump(exclude_none=True))

    return response_message.content or ""


def get_last_tool_errors() -> list[str]:
    """Return any {"error": ...} results from the most recent tool calls, for debugging."""
    errors: list[str] = []
    for message in st.session_state.messages[-6:]:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(result, dict) and "error" in result:
            errors.append(f"{message.get('name', 'tool')}: {result['error']}")
    return errors


def render_chat_history() -> None:
    """Render all user/assistant messages (skipping system/tool messages)."""
    for message in st.session_state.messages:
        if message["role"] in ("user", "assistant") and message.get("content"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="Weather AI Agent", page_icon="\U0001F324️")
    st.title("\U0001F324️ Weather AI Agent")
    st.caption(
        "Ask about the weather anywhere in the world. Powered by Groq "
        "function calling and the free Open-Meteo API (no weather API key "
        "needed)."
    )

    with st.sidebar:
        st.header("Settings")
        if DEFAULT_API_KEY:
            st.success("Using the built-in API key -- no setup needed.")
            api_key = DEFAULT_API_KEY
            with st.expander("Use a different key instead"):
                override_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Overrides the built-in key for this session only.",
                )
                if override_key:
                    api_key = override_key
        else:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                help=(
                    "Get a free key at https://console.groq.com/keys. It is "
                    "used only for this session and is never stored."
                ),
            )
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

    init_session_state()
    render_chat_history()

    user_input = st.chat_input("Ask about the weather, e.g. 'What's it like in Tokyo?'")
    if not user_input:
        return

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    client = get_client(api_key)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply_text = get_assistant_response(client)
            st.markdown(reply_text or "(no response)")

            tool_errors = get_last_tool_errors()
            if tool_errors:
                with st.expander("Debug: tool call failed"):
                    for tool_error in tool_errors:
                        st.code(tool_error)
        except AuthenticationError:
            st.error("Invalid Groq API key. Please check the key in the sidebar.")
        except RateLimitError as exc:
            st.error(
                "Groq's rate limit was hit. Please wait a moment and try "
                f"again. Details: {exc}"
            )
        except InternalServerError as exc:
            st.error(
                "Groq's servers are temporarily unavailable (this is on "
                f"their side, not this app). Please try again shortly. Details: {exc}"
            )
        except APIError as exc:
            st.error(f"Groq API error: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
