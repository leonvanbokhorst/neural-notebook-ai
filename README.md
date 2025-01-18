# Neural Notebook AI

A collection of proof-of-concepts (POCs) exploring AI agent architectures, LLM integration, and complex behavioral simulations.

## Proof of Concepts

### Voice Blending (pocs/voice_blending)

A sophisticated voice synthesis system using Kokoro TTS that demonstrates advanced voice blending capabilities:

- **Voice Embedding Manipulation**

  - Linear interpolation between voice embeddings
  - Spherical linear interpolation (SLERP) for natural transitions
  - Multi-voice blending (up to 3-way combinations)
  - Special blend combinations (e.g., British quartet, Transatlantic blend)

- **Technical Features**

  - Automatic model and voice file management
  - Dynamic voice loading and caching
  - Multi-platform support with espeak-ng integration
  - Configurable sample rate and device selection (CPU/CUDA/MPS)

- **Voice Library**

  - American and British accent variations
  - Male and female voice options
  - Cross-gender voice blending capabilities
  - Customizable voice combinations via YAML config

- **Output Generation**
  - High-quality 24kHz audio output
  - WAV file generation with proper normalization
  - Organized output directory structure
  - Detailed logging and error handling

### LiteLLM Integration (pocs/litellm)

A demonstration of using LiteLLM to interact with various LLM models, particularly focusing on Ollama integration. Features:

- Model-agnostic LLM interactions
- Usage tracking and statistics
- Streaming responses
- Configurable model parameters
- Callback handling for monitoring

### Weather Analysis Agent (pocs/pydantic_weather_agent)

An agentic weather service built using Pydantic.ai, demonstrating:

- Asynchronous weather data fetching
- Geocoding integration
- Structured data handling with Pydantic
- Tool-based agent architecture
- Streamlit-based visualization

## Development Standards

This project follows strict development standards including:

- PEP 8 style guide compliance
- Type hints (PEP 484)
- Comprehensive documentation
- Test coverage requirements
- Code review standards
- Error handling best practices

## Getting Started

Each POC contains its own README with specific setup instructions and requirements.

## Made with ❤️ and Human-Machine Collaboration

<img src="assets/cyborg.png" alt="Cyborg Logo" width="50%"/>

This project represents the fusion of human creativity and machine intelligence, working together to push the boundaries of what's possible.

Logo design by Chris Geene and Pieter Wels - [Welgeen Creative Studio](https://welgeen.nl/mmmlabel/)

