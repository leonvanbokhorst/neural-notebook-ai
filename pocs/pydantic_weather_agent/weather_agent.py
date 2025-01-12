"""
Weather Agent POC using Pydantic.ai
----------------------------------

A demonstration of building an agentic weather service using Pydantic.ai.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from httpx import AsyncClient
from pydantic_ai import Agent, ModelRetry, RunContext

# Load environment variables
load_dotenv()


@dataclass
class Dependencies:
    """Dependencies for the weather agent."""

    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


# Initialize the weather agent
weather_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "Be concise, reply with one sentence. "
        "Use get_lat_lng for coords, then get_weather for conditions."
    ),
    deps_type=Dependencies,
    retries=2,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Dependencies],
    loc: str,
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The run context with dependencies
        loc: Description of the location
    """
    if ctx.deps.geo_api_key is None:
        # Return dummy data if no API key (London coordinates)
        return {"lat": 51.5074, "lng": -0.1278}

    params = {
        "q": loc,
        "api_key": ctx.deps.geo_api_key,
    }

    r = await ctx.deps.client.get("https://geocode.maps.co/search", params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        return {"lat": float(data[0]["lat"]), "lng": float(data[0]["lon"])}
    raise ModelRetry("Location not found")


@weather_agent.tool
async def get_weather(
    ctx: RunContext[Dependencies],
    lat: float,
    lng: float,
) -> dict[str, Any]:
    """Get the weather at a specific location.

    Args:
        ctx: The run context with dependencies
        lat: Latitude of the location
        lng: Longitude of the location
    """
    if ctx.deps.weather_api_key is None:
        # Return dummy data if no API key
        return {"temperature": "21°C", "description": "Sunny"}

    params = {
        "apikey": ctx.deps.weather_api_key,
        "location": f"{lat},{lng}",
        "units": "metric",
    }

    r = await ctx.deps.client.get(
        "https://api.tomorrow.io/v4/weather/realtime",
        params=params,
    )
    r.raise_for_status()
    data = r.json()

    values = data["data"]["values"]
    weather_codes = {
        1000: "Clear, Sunny",
        1100: "Mostly Clear",
        1101: "Partly Cloudy",
        1102: "Mostly Cloudy",
        1001: "Cloudy",
        4000: "Drizzle",
        4001: "Rain",
        4200: "Light Rain",
        4201: "Heavy Rain",
        5000: "Snow",
        5100: "Light Snow",
        5101: "Heavy Snow",
        8000: "Thunderstorm",
    }

    return {
        "temperature": f"{values['temperature']:0.1f}°C",
        "description": weather_codes.get(values["weatherCode"], "Unknown"),
    }


async def query_weather(location: str) -> str:
    """Query the weather for a specific location.

    Args:
        location: The location to get weather for
    """
    async with AsyncClient() as client:
        deps = Dependencies(
            client=client,
            weather_api_key=os.getenv("WEATHER_API_KEY"),
            geo_api_key=os.getenv("GEO_API_KEY"),
        )

        result = await weather_agent.run(
            f"What is the weather like in {location}?",
            deps=deps,
        )
        return result.data


async def main():
    """Main entry point for demonstration."""
    location = input("Enter location: ")
    response = await query_weather(location)
    print(f"Weather: {response}")


if __name__ == "__main__":
    asyncio.run(main())
