"""
Streamlit UI for Weather Analysis System
--------------------------------------

A web interface for weather analysis and insights.
"""

import asyncio

import streamlit as st
from weather_agent import query_weather
from weather_analyst import analyze_weather

st.set_page_config(
    page_title="Weather Analysis System",
    page_icon="🌤️",
    layout="wide",
)

st.title("🌤️ Weather Analysis System")
st.markdown(
    "This demo shows how to use Pydantic.ai to create an intelligent "
    "weather analysis system that understands natural language."
)

# Sidebar for analysis options
with st.sidebar:
    st.header("Analysis Options")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["current", "comparative", "anomaly"],
        format_func=lambda x: x.title(),
    )

    # Multi-location input
    st.subheader("Locations")
    num_locations = st.number_input(
        "Number of locations",
        1,
        5,
        1,
    )
    locations = []
    for i in range(num_locations):
        loc = st.text_input(
            f"Location {i+1}",
            placeholder="e.g., London, Tokyo",
            key=f"location_{i}",
        )
        if loc:
            locations.append(loc)

# Main content area
if locations:
    with st.spinner("Analyzing weather patterns..."):
        try:
            if len(locations) == 1:
                # Single location query
                response = asyncio.run(query_weather(locations[0]))
                st.success("Weather information retrieved!")
                st.info(response)
            else:
                # Multi-location analysis
                response = asyncio.run(analyze_weather(locations, analysis_type))
                st.success("Analysis complete!")

                # Display results based on analysis type
                if analysis_type == "comparative":
                    st.subheader("Temperature Comparison")
                    for loc in locations:
                        st.write(f"📍 {loc}")
                    st.info(response)
                elif analysis_type == "anomaly":
                    st.subheader("Weather Anomalies")
                    st.warning(response)
                else:
                    st.subheader("Current Conditions")
                    st.info(response)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Add usage instructions
with st.expander("How to use"):
    st.markdown(
        """
    1. Select analysis type from the sidebar
    2. Enter one or more locations
    3. The system will:
        - Gather weather data for all locations
        - Perform the selected analysis
        - Provide insights and comparisons

    Analysis Types:
    - **Current**: Simple weather report
    - **Comparative**: Compare weather between locations
    - **Anomaly**: Detect unusual weather patterns
    """
    )

# Add API key instructions
with st.expander("API Keys Setup"):
    st.markdown(
        """
    To use real weather data, you'll need:
    1. A weather API key from [Tomorrow.io](https://www.tomorrow.io)
    2. A geocoding API key from [Geocode.maps.co](https://geocode.maps.co)
    3. An OpenAI API key for the analysis

    Set these in your `.env` file as:
    ```
    WEATHER_API_KEY=your_key_here
    GEO_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here
    ```
    """
    )
