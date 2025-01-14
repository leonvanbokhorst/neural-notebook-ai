# Pydantic.ai Weather Agent POC

A demonstration of building an intelligent weather agent using Pydantic.ai. This POC showcases how to create an agentic system that can understand natural language queries about weather conditions and provide relevant responses.

## Overview

This POC demonstrates:

- Building agents with Pydantic.ai
- Tool-based implementations for geocoding and weather data
- Async operations with HTTPX
- Streamlit-based UI for interaction

## Key Concepts

- **Pydantic.ai Agents**: Using the Agent class to create intelligent systems
- **Tool Implementation**: Creating tools for location and weather data retrieval
- **Dependency Injection**: Managing API clients and keys
- **Error Handling**: Graceful fallbacks and retries

## Implementation Details

The POC consists of two main components:

1. **Weather Agent** (`weather_agent.py`):

   - Implements the core agent functionality
   - Provides tools for geocoding and weather data
   - Handles API interactions and data processing

2. **Streamlit UI** (`streamlit_app.py`):
   - Provides a web interface for interaction
   - Displays weather information
   - Includes usage instructions

## Requirements

```
# Install dependencies
pip install -r requirements.txt

# Set up environment variables in .env:
WEATHER_API_KEY=your_tomorrow_io_key
GEO_API_KEY=your_geocode_maps_key
```

## Usage

1. Run the command-line version:

```bash
python weather_agent.py
```

2. Run the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Results and Visualization

The agent can handle queries like:

- "What's the weather in London?"
- "How's the weather in Paris and Rome?"
- "Current weather in Tokyo?"

Results include:

- Temperature in Celsius
- Weather description
- Natural language response

## References

- [Pydantic.ai Documentation](https://ai.pydantic.dev/)
- [Weather API (Tomorrow.io)](https://www.tomorrow.io/)
- [Geocoding API (geocode.maps.co)](https://geocode.maps.co/)

## Notes

- The system falls back to dummy data if API keys are not provided
- Supports multiple location queries in a single request
- Includes retry logic for failed API calls

## License

MIT License
