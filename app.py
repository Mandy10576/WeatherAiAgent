"""
app.py

A two-agent Streamlit app powered by Groq (OpenAI-compatible API):

  * Agent 1 -- Weather & Style Agent: a chat agent using Groq function
    calling plus the free, key-less Open-Meteo APIs for real weather data.
  * Agent 2 -- Dress & Fashion Finder Agent: refines a shopper's request into
    high-intent e-commerce keywords with the LLM, then pulls real product
    results (title, image, buy link) from DuckDuckGo.

Pick an agent from the sidebar selector.

Dependencies (see requirements.txt):
    streamlit
    openai
    requests
    python-dotenv
    ddgs

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from fashion_tool import search_products_safe
from pdf_tool import (
    PdfProcessingError,
    build_index_from_pdf,
    is_broad_query,
    sample_chunks,
    search as search_pdf_index,
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


# ---------------------------------------------------------------------------
# Agent 1 -- Weather & Style Agent
# ---------------------------------------------------------------------------


def render_weather_agent(api_key: str) -> None:
    """Render the original weather chat agent (logic unchanged)."""
    st.title("\U0001F324\ufe0f Weather AI Agent")
    st.caption(
        "Ask about the weather anywhere in the world. Powered by Groq "
        "function calling and the free Open-Meteo API (no weather API key "
        "needed)."
    )

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


# ---------------------------------------------------------------------------
# Agent 2 -- Dress & Fashion Finder Agent
# ---------------------------------------------------------------------------

FASHION_SYSTEM_PROMPT = (
    "You rewrite a shopper's request into a short, high-intent e-commerce "
    "search keyword string. Reply with ONLY that keyword string -- no "
    "quotes, no explanation, no bullet points. Keep it under 12 words and "
    "preserve every garment type, colour, fabric, fit, occasion, gender and "
    "size cue the shopper gave, word for word where possible. Do NOT invent "
    "or add a gender ('women', 'men', 'unisex'), size, fit, or occasion that "
    "the shopper did not state or clearly imply -- if they said 'black coat', "
    "keep it exactly as 'black coat', not 'women black coat'. If the request "
    "is vague on the garment itself, add the most likely garment noun only. "
    "Example: 'something sparkly for my friend's evening wedding reception' "
    "-> 'sequin embellished evening gown wedding reception'."
)
FASHION_REFINE_MAX_TOKENS = 200
FASHION_RESULT_COUNT = 9
# Fetch extra candidates so that dropping ones with a dead image still
# leaves close to a full grid.
FASHION_SEARCH_FETCH_COUNT = 18
FASHION_GRID_COLUMNS = 3
FASHION_TITLE_MAX_CHARS = 70


def refine_search_keywords(client: OpenAI, raw_query: str) -> str:
    """
    Use the LLM to turn a natural-language request into search keywords.

    Falls back to the shopper's own words if the model returns nothing usable,
    so a flaky refinement never blocks the actual product search.
    """
    messages = [
        {"role": "system", "content": FASHION_SYSTEM_PROMPT},
        {"role": "user", "content": raw_query},
    ]
    try:
        # gpt-oss is a reasoning model: without a low effort setting and a
        # generous token budget the reasoning trace eats the whole completion
        # and `content` comes back empty.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=FASHION_REFINE_MAX_TOKENS,
            reasoning_effort="low",
        )
    except BadRequestError:
        # Models that don't accept `reasoning_effort` still work without it.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=FASHION_REFINE_MAX_TOKENS,
        )

    keywords = (response.choices[0].message.content or "").strip().strip('"')
    return keywords or raw_query


def truncate_title(title: str) -> str:
    """Shorten a product title so grid cards stay a consistent height."""
    clean = " ".join(title.split())
    if len(clean) <= FASHION_TITLE_MAX_CHARS:
        return clean
    return clean[: FASHION_TITLE_MAX_CHARS - 1].rstrip() + "\u2026"


IMAGE_FETCH_TIMEOUT_SECONDS = 4
IMAGE_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_image_bytes(image_url: str) -> Optional[bytes]:
    """
    Fetch product image bytes server-side so a dead/blocked link never shows
    up as Streamlit's raw broken-image icon.

    `st.image(url)` just emits an `<img src="...">` tag, so it can't tell a
    real image apart from a 404 or a hotlink block -- both render as the
    browser's placeholder. Fetching the bytes ourselves (with a browser-like
    User-Agent and a same-site Referer) lets us fall back to a clean caption
    instead. Cached per URL so re-rendering the same grid stays fast.
    """
    try:
        domain = urlparse(image_url).netloc
        headers = dict(IMAGE_FETCH_HEADERS)
        if domain:
            headers["Referer"] = f"https://{domain}/"
        response = requests.get(
            image_url, headers=headers, timeout=IMAGE_FETCH_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            return None
        return response.content
    except requests.RequestException:
        return None


def prefetch_images(products: list[dict[str, str]]) -> dict[str, Optional[bytes]]:
    """Fetch the image bytes for all products in parallel; returns url -> bytes."""
    urls = [product["image_url"] for product in products]
    with ThreadPoolExecutor(max_workers=min(8, len(urls) or 1)) as pool:
        fetched = list(pool.map(fetch_image_bytes, urls))
    return dict(zip(urls, fetched))


def keep_products_with_images(
    products: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    """Drop products whose image failed to load; a listing with no picture isn't useful."""
    image_bytes_by_url = prefetch_images(products)
    return [
        product
        for product in products
        if image_bytes_by_url.get(product["image_url"])
    ][:limit]


def render_product_grid(products: list[dict[str, str]]) -> None:
    """Render products as a responsive grid of image + title + buy link cards."""
    for row_start in range(0, len(products), FASHION_GRID_COLUMNS):
        row = products[row_start : row_start + FASHION_GRID_COLUMNS]
        columns = st.columns(FASHION_GRID_COLUMNS)
        for column, product in zip(columns, row):
            with column:
                st.image(fetch_image_bytes(product["image_url"]), width="stretch")
                st.markdown(f"**{truncate_title(product['title'])}**")
                if product.get("source"):
                    st.caption(product["source"])
                st.markdown(f"[\U0001F6D2 **Buy now**]({product['product_url']})")
                st.write("")


def run_fashion_search(api_key: str, raw_query: str) -> None:
    """Refine the query, search for products, and stash the outcome in session state."""
    client = get_client(api_key)

    try:
        with st.spinner("Optimizing your search with the LLM..."):
            keywords = refine_search_keywords(client, raw_query)
    except AuthenticationError:
        st.error("Invalid Groq API key. Please check the key in the sidebar.")
        return
    except (RateLimitError, InternalServerError, APIError) as exc:
        st.warning(
            f"Could not refine the query ({exc}); searching your words as typed."
        )
        keywords = raw_query

    with st.spinner(f"Searching the web for '{keywords}'..."):
        products, error = search_products_safe(
            keywords, max_results=FASHION_SEARCH_FETCH_COUNT
        )

    if not error and products:
        with st.spinner("Checking product images..."):
            products = keep_products_with_images(products, FASHION_RESULT_COUNT)

    st.session_state.fashion_results = {
        "raw_query": raw_query,
        "keywords": keywords,
        "products": products,
        "error": error,
    }


def render_fashion_agent(api_key: str) -> None:
    """Render the dress/fashion product finder agent."""
    st.title("\U0001F457 Dress & Fashion Finder Agent")
    st.caption(
        "Describe what you want to wear in plain English. The LLM turns it "
        "into an e-commerce search, then the agent pulls real products off "
        "the web with buy links."
    )

    with st.form("fashion_search_form"):
        raw_query = st.text_input(
            "What are you looking for?",
            placeholder=(
                "e.g. red velvet dress, black formal suit, oversized graphic hoodie"
            ),
        )
        submitted = st.form_submit_button("Find outfits")

    if submitted:
        if not raw_query.strip():
            st.warning("Please describe the item you're looking for.")
        elif not api_key:
            st.warning("Please enter your Groq API key in the sidebar to continue.")
        else:
            run_fashion_search(api_key, raw_query.strip())

    results = st.session_state.get("fashion_results")
    if not results:
        return

    if results["error"]:
        st.error(f"Search failed: {results['error']}")
        return

    if not results["products"]:
        st.info(
            f"No products found for '{results['keywords']}'. Try different "
            "wording, a broader colour, or a more common garment name."
        )
        return

    st.success(f"Optimized search: `{results['keywords']}`")
    render_product_grid(results["products"])


# ---------------------------------------------------------------------------
# Agent 3 -- PDF Q&A Agent
# ---------------------------------------------------------------------------

PDF_SYSTEM_PROMPT_TEMPLATE = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "excerpts from their PDF given below as context -- do not use outside "
    "knowledge. If the excerpts don't contain the answer, say clearly that "
    "the document doesn't seem to cover that, instead of guessing.\n\n"
    "--- PDF excerpts ---\n{context}\n--- end of excerpts ---"
)
PDF_TOP_K_CHUNKS = 4
PDF_BROAD_SAMPLE_CHUNKS = 6
PDF_ANSWER_MAX_TOKENS = 500


def process_uploaded_pdf(uploaded_file: Any) -> None:
    """Extract, chunk, and index an uploaded PDF; store the index in session state."""
    try:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            index = build_index_from_pdf(uploaded_file)
    except PdfProcessingError as exc:
        st.session_state.pdf_index = None
        st.session_state.pdf_error = str(exc)
        return

    st.session_state.pdf_index = index
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.pdf_error = None
    st.session_state.pdf_qa_history = []


def answer_from_pdf(client: OpenAI, question: str) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and ask the LLM to answer using only them."""
    index = st.session_state.pdf_index

    if is_broad_query(question):
        # Overview-style questions ("what's in this pdf?", "summarize it")
        # rarely share keywords with the document's own body text, so
        # keyword-overlap search would score everything near zero. Give the
        # LLM a spread of chunks across the whole document instead.
        context_chunks = sample_chunks(index, max_chunks=PDF_BROAD_SAMPLE_CHUNKS)
    else:
        matches = search_pdf_index(index, question, top_k=PDF_TOP_K_CHUNKS)
        if not matches:
            return (
                "I couldn't find anything in the document related to that "
                "question. Try rephrasing it or asking about a topic the PDF "
                "actually covers.",
                [],
            )
        context_chunks = [chunk for chunk, _score in matches]

    context = "\n\n".join(
        f"[Excerpt {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    system_prompt = PDF_SYSTEM_PROMPT_TEMPLATE.format(context=context)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=PDF_ANSWER_MAX_TOKENS,
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer or "The model returned an empty response -- please try again.", context_chunks


def render_pdf_agent(api_key: str) -> None:
    """Render the PDF Q&A agent: upload a PDF, then ask questions about it."""
    st.title("\U0001F4C4 PDF Q&A Agent")
    st.caption(
        "Upload a PDF, then ask questions about it. The agent retrieves the "
        "most relevant parts of the document and answers using only that "
        "content."
    )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None and st.session_state.get("pdf_name") != uploaded_file.name:
        process_uploaded_pdf(uploaded_file)

    if st.session_state.get("pdf_error"):
        st.error(st.session_state.pdf_error)
        return

    if not st.session_state.get("pdf_index"):
        st.info("Upload a PDF above to start asking questions about it.")
        return

    st.success(f"Loaded **{st.session_state.pdf_name}** -- ready for questions.")

    for entry in st.session_state.get("pdf_qa_history", []):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry["sources"]:
                with st.expander("Sources used from the PDF"):
                    for i, chunk in enumerate(entry["sources"]):
                        st.caption(f"Excerpt {i + 1}")
                        st.text(chunk)

    question = st.chat_input("Ask a question about the uploaded PDF...")
    if not question:
        return

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    with st.chat_message("user"):
        st.markdown(question)

    client = get_client(api_key)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document and thinking..."):
                answer, sources = answer_from_pdf(client, question)
            st.markdown(answer)
            if sources:
                with st.expander("Sources used from the PDF"):
                    for i, chunk in enumerate(sources):
                        st.caption(f"Excerpt {i + 1}")
                        st.text(chunk)
        except AuthenticationError:
            st.error("Invalid Groq API key. Please check the key in the sidebar.")
            return
        except RateLimitError as exc:
            st.error(f"Groq's rate limit was hit. Please wait a moment and try again. Details: {exc}")
            return
        except InternalServerError as exc:
            st.error(f"Groq's servers are temporarily unavailable. Please try again shortly. Details: {exc}")
            return
        except APIError as exc:
            st.error(f"Groq API error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
            st.error(f"Unexpected error: {exc}")
            return

    st.session_state.setdefault("pdf_qa_history", []).append(
        {"question": question, "answer": answer, "sources": sources}
    )


# ---------------------------------------------------------------------------
# Shell -- sidebar navigation shared by all agents
# ---------------------------------------------------------------------------

WEATHER_AGENT = "\U0001F324\ufe0f Agent 1: Weather & Style"
FASHION_AGENT = "\U0001F457 Agent 2: Dress & Fashion Finder"
PDF_AGENT = "\U0001F4C4 Agent 3: PDF Q&A"


def render_sidebar() -> tuple[str, str]:
    """Render the agent selector plus shared settings; return (agent, api_key)."""
    with st.sidebar:
        st.header("Agents")
        selected_agent = st.radio(
            "Choose an agent",
            (WEATHER_AGENT, FASHION_AGENT, PDF_AGENT),
            label_visibility="collapsed",
        )

        st.markdown("---")
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
        if selected_agent == WEATHER_AGENT:
            if st.button("Clear conversation"):
                st.session_state.messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                st.rerun()
        elif selected_agent == FASHION_AGENT:
            if st.button("Clear results"):
                st.session_state.pop("fashion_results", None)
                st.rerun()
        elif st.button("Clear PDF"):
            for key in ("pdf_index", "pdf_name", "pdf_error", "pdf_qa_history"):
                st.session_state.pop(key, None)
            st.rerun()

    return selected_agent, api_key


def main() -> None:
    st.set_page_config(page_title="Multi-Agent AI Studio", page_icon="\U0001F916")

    selected_agent, api_key = render_sidebar()

    if selected_agent == FASHION_AGENT:
        render_fashion_agent(api_key)
    elif selected_agent == PDF_AGENT:
        render_pdf_agent(api_key)
    else:
        render_weather_agent(api_key)


if __name__ == "__main__":
    main()
