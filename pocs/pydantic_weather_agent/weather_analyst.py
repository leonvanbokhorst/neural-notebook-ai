"""
Weather Analyst Agent
-------------------

An intelligent agent that analyzes weather patterns and provides insights.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

from httpx import AsyncClient
from pydantic_ai import Agent, RunContext
from weather_agent import Dependencies, get_lat_lng, get_weather


@dataclass
class WeatherPattern:
    """A weather pattern with location and conditions."""

    location: str
    temperature: float
    description: str
    timestamp: datetime
    is_anomaly: bool = False


# Initialize the analyst agent with more complex reasoning capabilities
analyst_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a weather expert. Analyze patterns and provide insights. "
        "Use tools to gather data and identify patterns. "
        "Consider norms, location, and time. "
        "Be specific about unusual patterns."
    ),
    deps_type=Dependencies,
    retries=2,
)


@analyst_agent.tool
async def analyze_location_weather(
    ctx: RunContext[Dependencies],
    locations: List[str],
    analysis_type: str = "current",
) -> dict[str, Any]:
    """Analyze weather patterns for multiple locations.

    Args:
        ctx: The run context
        locations: List of locations to analyze
        analysis_type: Type of analysis to perform
    """
    patterns = []

    for location in locations:
        coords = await get_lat_lng(ctx, location)
        weather = await get_weather(ctx, coords["lat"], coords["lng"])

        pattern = WeatherPattern(
            location=location,
            temperature=float(weather["temperature"].replace("°C", "")),
            description=weather["description"],
            timestamp=datetime.now(),
        )

        if pattern.temperature > 30 or pattern.temperature < -10:
            pattern.is_anomaly = True

        patterns.append(pattern)

    if analysis_type == "comparative":
        temp_diff = max(p.temperature for p in patterns) - min(
            p.temperature for p in patterns
        )
        return {
            "patterns": [vars(p) for p in patterns],
            "temperature_range": f"{temp_diff:.1f}°C",
            "analysis_type": "comparative",
        }
    elif analysis_type == "anomaly":
        anomalies = [p for p in patterns if p.is_anomaly]
        return {
            "patterns": [vars(p) for p in patterns],
            "anomalies": [vars(p) for p in anomalies],
            "analysis_type": "anomaly",
        }
    else:
        return {
            "patterns": [vars(p) for p in patterns],
            "analysis_type": "current",
        }


async def analyze_weather(locations: List[str], analysis_type: str = "current") -> str:
    """Analyze weather patterns for given locations.

    Args:
        locations: List of locations to analyze
        analysis_type: Type of analysis to perform
    """
    async with AsyncClient() as client:
        deps = Dependencies(
            client=client,
            weather_api_key=os.getenv("WEATHER_API_KEY"),
            geo_api_key=os.getenv("GEO_API_KEY"),
        )

        # Create analysis prompt based on type
        if analysis_type == "comparative":
            prompt = f"Compare weather in: {', '.join(locations)}"
        elif analysis_type == "anomaly":
            prompt = f"Check weather anomalies in: {', '.join(locations)}"
        else:
            prompt = f"Analyze weather in: {', '.join(locations)}"

        result = await analyst_agent.run(prompt, deps=deps)
        return result.data


async def main():
    """Main entry point for demonstration."""
    locations = ["London", "Tokyo", "New York", "Sydney"]

    print("\nCurrent Weather Analysis:")
    response = await analyze_weather(locations, "current")
    print(response)

    print("\nComparative Analysis:")
    response = await analyze_weather(locations, "comparative")
    print(response)

    print("\nAnomaly Detection:")
    response = await analyze_weather(locations, "anomaly")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
