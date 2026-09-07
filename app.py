"""
app.py

Weather AI Agent -- a Streamlit chat app powered by Google Gemini's
function calling and the free, key-less Open-Meteo APIs for real weather data.

Dependencies (see requirements.txt):
    streamlit
    google-genai
    requests
    python-dotenv

Get a free Gemini API key at: https://aistudio.google.com/apikey
Set it as GEMINI_API_KEY in a local .env file to avoid asking every visitor
for their own key; the sidebar field remains as an optional override.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from weather_tool import get_current_weather

load_dotenv()
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME = "gemini-3.6-flash"
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable weather assistant. Use the "
    "get_current_weather tool whenever the user asks about weather, "
    "temperature, or conditions in a specific place. Answer follow-up "
    "questions (e.g. what to wear, whether to bring an umbrella) using the "
    "most recent weather data already retrieved in this conversation "
    "whenever possible, only calling the tool again if the user asks about "
    "a new location or wants a fresh reading."
)


def init_session_state() -> None:
    """Initialize Streamlit session state on first run."""
    if "chat_messages" not in st.session_state:
        # Plain (role, text) pairs used only for rendering the chat history.
        st.session_state.chat_messages = []
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    if "chat_client" not in st.session_state:
        # Must be kept alive in session_state: once this Client object is
        # garbage collected, its underlying HTTP client closes and any chat
        # session created from it stops working ("client has been closed").
        st.session_state.chat_client = None
    if "chat_api_key" not in st.session_state:
        st.session_state.chat_api_key = None


def get_or_create_chat_session(api_key: str) -> genai.chats.Chat:
    """Return the active Gemini chat session, recreating it if the key changed."""
    if st.session_state.chat_session is None or st.session_state.chat_api_key != api_key:
        client = genai.Client(api_key=api_key)
        st.session_state.chat_client = client
        st.session_state.chat_session = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[get_current_weather],
            ),
        )
        st.session_state.chat_api_key = api_key
        st.session_state.chat_messages = []
    return st.session_state.chat_session


def get_last_tool_errors(chat: genai.chats.Chat) -> list[str]:
    """Inspect the chat history for tool calls that returned an "error" key.

    Gemini's automatic function calling swallows any exception raised while
    running a tool and feeds the model {"error": "..."} instead, so the
    model's reply alone never reveals what actually went wrong. Surfacing
    this separately lets the UI show the real cause instead of the model's
    vague paraphrase.
    """
    errors: list[str] = []
    for content in chat.get_history(curated=False)[-4:]:
        for part in content.parts or []:
            if not part.function_response:
                continue
            response = part.function_response.response or {}
            tool_result = response.get("result", response)
            if isinstance(tool_result, dict) and "error" in tool_result:
                errors.append(f"{part.function_response.name}: {tool_result['error']}")
            elif "error" in response:
                errors.append(f"{part.function_response.name}: {response['error']}")
    return errors


def render_chat_history() -> None:
    """Render all stored user/assistant turns."""
    for role, text in st.session_state.chat_messages:
        with st.chat_message(role):
            st.markdown(text)


def main() -> None:
    st.set_page_config(page_title="Weather AI Agent", page_icon="\U0001F324️")
    st.title("\U0001F324️ Weather AI Agent")
    st.caption(
        "Ask about the weather anywhere in the world. Powered by Google "
        "Gemini function calling and the free Open-Meteo API (no weather "
        "API key needed)."
    )

    with st.sidebar:
        st.header("Settings")
        if DEFAULT_API_KEY:
            st.success("Using the built-in API key -- no setup needed.")
            api_key = DEFAULT_API_KEY
            with st.expander("Use a different key instead"):
                override_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Overrides the built-in key for this session only.",
                )
                if override_key:
                    api_key = override_key
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help=(
                    "Get a free key at https://aistudio.google.com/apikey. "
                    "It is used only for this session and is never stored."
                ),
            )
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.chat_session = None
            st.session_state.chat_client = None
            st.session_state.chat_messages = []
            st.rerun()

    init_session_state()
    render_chat_history()

    user_input = st.chat_input("Ask about the weather, e.g. 'What's it like in Tokyo?'")
    if not user_input:
        return

    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to continue.")
        return

    st.session_state.chat_messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            chat = get_or_create_chat_session(api_key)
            with st.spinner("Thinking..."):
                response = chat.send_message(user_input)
            reply_text = response.text or "(no response)"
            st.markdown(reply_text)
            st.session_state.chat_messages.append(("assistant", reply_text))

            tool_errors = get_last_tool_errors(chat)
            if tool_errors:
                with st.expander("Debug: tool call failed"):
                    for tool_error in tool_errors:
                        st.code(tool_error)
        except ClientError as exc:
            st.error(
                "Gemini rejected the request -- check that your API key is "
                f"valid and has quota remaining. Details: {exc}"
            )
        except APIError as exc:
            st.error(f"Gemini API error: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
