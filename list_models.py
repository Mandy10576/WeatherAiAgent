"""One-off helper: lists Gemini models available to your API key."""
import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY") or input("Paste your Gemini API key: ").strip()
client = genai.Client(api_key=api_key)
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)
