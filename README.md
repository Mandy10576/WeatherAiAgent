# Weather & Fashion AI Agents — Interview Prep Notes

Quick-revision notes for the Weather AI Agent (Agent 1). Read top to bottom in ~5 minutes before an interview.

## 1. One-liner

A Streamlit chat app where an LLM (hosted on Groq, OpenAI-compatible API) uses **function calling** to fetch real-time weather for any city, with no weather API key required.

## 2. Tech stack — what and why

| Piece | Used | Why |
|---|---|---|
| UI | **Streamlit** | Python-only, fast to build, `st.chat_message` / `st.chat_input` give a ready-made chat UI — no separate frontend needed |
| LLM provider | **Groq API** (OpenAI-compatible endpoint) | Fast free-tier inference (Groq's LPU hardware); the existing `openai` Python SDK works unchanged — only `base_url` points at Groq |
| Model | `openai/gpt-oss-120b` | Open-weight reasoning model hosted by Groq, supports tool/function calling |
| Weather data | **Open-Meteo API** (Geocoding + Forecast) | Completely free, **no API key at all** — the main differentiator vs. OpenWeatherMap-style APIs |
| Secrets | `python-dotenv` + `.env` | Key loaded from environment, never hardcoded; `.env` is git-ignored |

**Likely follow-up:** *"Why not OpenAI directly?"*
→ Groq's free tier is fast and reliable, and since its API is OpenAI-compatible, only `base_url` changes — swapping providers later is a one-line change.

## 3. Why two API calls for weather (commonly asked)

Open-Meteo works in two steps:
1. **Geocoding API** (`_geocode_location`) — turns a city name ("Tokyo") into **latitude/longitude**.
2. **Forecast API** (`_fetch_forecast`) — takes those coordinates and returns current temperature, humidity, wind, precipitation, weather code.

So a single weather lookup is two sequential HTTP calls: name → coordinates → conditions.

## 4. Function-calling flow (the core concept — know this cold)

1. User message → appended to `st.session_state.messages` (holds the whole conversation).
2. `WEATHER_TOOL_SCHEMA` is a JSON schema describing a `get_current_weather(location)` tool — its purpose and required parameter — sent to the model.
3. `client.chat.completions.create(..., tools=[WEATHER_TOOL_SCHEMA], tool_choice="auto")` — the model itself decides whether the query needs the tool.
4. If the model requests a tool call (`response_message.tool_calls`), `run_tool_call()` executes the real Python function `get_current_weather()`.
5. The tool's result goes back into the conversation as a `{"role": "tool", "content": ...}` message.
6. The model is called again so it can turn that raw data into a natural-language answer.

This is a **loop** (`while response_message.tool_calls`), because the model can occasionally chain multiple tool calls. It's capped at `MAX_TOOL_CALLS_PER_MESSAGE = 2` — if a location is ambiguous, this stops the model from retrying endlessly and racking up API calls.

## 5. Reliability & error handling (shows seniority — say this confidently)

- **Retry with backoff** (`call_with_retry`): on `RateLimitError` (429) or `InternalServerError` (5xx), retries up to 3 times with increasing delay (2s, 4s, 6s).
- **Tool errors never crash the app**: `get_current_weather()` never raises outward — a bad city or API failure returns `{"error": "..."}`, so the model can react gracefully in conversation instead of the app breaking.
- **System prompt tells the model not to guess spellings**: if a location isn't found, it must ask the user to clarify rather than silently retrying variant spellings — saves wasted tool calls.
- **UI-level try/except** distinguishes `AuthenticationError`, `RateLimitError`, `InternalServerError`, `APIError`, and a generic fallback — each gets its own user-facing message.

## 6. Session state

Streamlit **reruns the entire script top to bottom** on every interaction — this is the concept to explain if asked "what is a Streamlit rerun?"
- `st.session_state.messages` persists the chat history across reruns.
- `init_session_state()` seeds the system prompt only on first run.
- "Clear conversation" resets session state and calls `st.rerun()`.

## 7. Security practice

- API key loaded via `os.environ.get("GROQ_API_KEY")` from `.env` — never hardcoded.
- `.env` is in `.gitignore` so the secret is never committed.
- Sidebar lets a user override with their own key if the built-in one is missing or rate-limited.

## 8. Q&A cheat sheet

**Q: Function calling vs. plain prompting?**
Plain prompting only generates text from training knowledge — no real-time data. Function calling lets the model request a structured tool call; we execute it, feed back real data, and the model turns that into a natural answer.

**Q: What if Open-Meteo goes down?**
`WeatherLookupError` is raised → caught inside `get_current_weather` → returned as `{"error": ...}` → the model tells the user data isn't available. The app itself never crashes.

**Q: How are follow-ups ("should I carry an umbrella?") handled without a new API call?**
The system prompt explicitly tells the model to reuse the most recently fetched weather data from conversation history, only calling the tool again for a new location or an explicit fresh reading — this avoids redundant tool calls and cost.

**Q: What does `tool_choice="auto"` mean?**
The model itself decides when a tool call is needed — it isn't forced on every message. A plain "hi" won't trigger a tool call.

---

*Agent 2 (Dress & Fashion Finder) was added later in the same app — sidebar `st.radio` switches between the two agents, sharing the same Groq client and API key handling. It uses the LLM to refine a free-text query into e-commerce search keywords, then `ddgs` (DuckDuckGo search) to fetch real product images/links, with server-side image validation so dead links are filtered out before rendering.*
