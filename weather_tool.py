"""
weather_tool.py

Self-contained weather lookup tool backed entirely by the free, key-less
Open-Meteo APIs:
  - Geocoding: https://geocoding-api.open-meteo.com/v1/search
  - Forecast:  https://api.open-meteo.com/v1/forecast

No API key of any kind is required to use this module.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10

# Open-Meteo returns WMO weather codes as integers; map the common ones to
# human-readable descriptions so the agent doesn't have to guess.
WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherLookupError(Exception):
    """Raised whenever a weather lookup cannot be completed."""


@dataclass(frozen=True)
class Location:
    """A resolved geocoded location."""

    name: str
    country: str
    latitude: float
    longitude: float

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


def _geocode_location(location: str) -> Location:
    """Convert a free-text city name into coordinates via Open-Meteo Geocoding.

    Raises:
        WeatherLookupError: if the request fails, times out, or no match is found.
    """
    if not location or not location.strip():
        raise WeatherLookupError("Location must be a non-empty string.")

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": location.strip(), "count": 1, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise WeatherLookupError(f"Geocoding request timed out for '{location}'.") from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherLookupError(f"Geocoding request failed for '{location}': {exc}") from exc

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise WeatherLookupError("Geocoding API returned an invalid response.") from exc

    results = payload.get("results") or []
    if not results:
        raise WeatherLookupError(f"Could not find a location matching '{location}'.")

    top_match = results[0]
    return Location(
        name=top_match.get("name", location),
        country=top_match.get("country", ""),
        latitude=top_match["latitude"],
        longitude=top_match["longitude"],
    )


def _fetch_forecast(location: Location) -> dict[str, Any]:
    """Fetch current weather conditions for a resolved location.

    Raises:
        WeatherLookupError: if the request fails, times out, or the response
            is missing the expected fields.
    """
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "wind_speed_10m",
                    "weather_code",
                ],
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise WeatherLookupError(
            f"Forecast request timed out for '{location.display_name}'."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherLookupError(
            f"Forecast request failed for '{location.display_name}': {exc}"
        ) from exc

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise WeatherLookupError("Forecast API returned an invalid response.") from exc

    current = payload.get("current")
    if not current:
        raise WeatherLookupError(
            f"Forecast API returned no current conditions for '{location.display_name}'."
        )
    return current


def get_current_weather(location: str) -> dict[str, Any]:
    """Look up current weather conditions for a city name.

    This is the function exposed to the Gemini model via function calling.
    It never raises to the caller for expected failure modes (bad city name,
    timeout, malformed API response) -- instead it returns a JSON-serializable
    dict with an "error" key so the model can react gracefully in conversation.

    Args:
        location: A free-text city name, e.g. "Paris", "San Francisco, CA".

    Returns:
        On success, a dict with keys: location, country, latitude, longitude,
        temperature_c, relative_humidity_pct, precipitation_mm,
        wind_speed_kmh, condition.
        On failure, a dict with a single "error" key describing the problem.
    """
    try:
        resolved_location = _geocode_location(location)
        current = _fetch_forecast(resolved_location)
    except WeatherLookupError as exc:
        return {"error": str(exc)}

    weather_code: Optional[int] = current.get("weather_code")
    condition = WEATHER_CODE_DESCRIPTIONS.get(weather_code, "Unknown")

    return {
        "location": resolved_location.name,
        "country": resolved_location.country,
        "latitude": resolved_location.latitude,
        "longitude": resolved_location.longitude,
        "temperature_c": current.get("temperature_2m"),
        "relative_humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "condition": condition,
    }
