# LiteLLM POC

This POC demonstrates the usage of LiteLLM to interact with multiple LLM providers through a unified interface.

## Features

- Unified interface for multiple LLM providers (OpenAI, Anthropic, Google)
- Configurable model parameters and API settings
- Error handling and logging
- Response comparison across different models

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Configure your API keys:
   - Copy the `config.yaml` file
   - Fill in your API keys for the providers you want to use:
     - OpenAI API key
     - Anthropic API key
     - Google Cloud API key

## Usage

1. Run the demonstration:

```bash
python litellm_demo.py
```

The script will:

- Load the configuration from `config.yaml`
- Send the same prompt to multiple LLM providers
- Compare and log the responses

## Configuration

Edit `config.yaml` to:

- Add/remove models
- Modify the prompt template
- Adjust token limits
- Configure logging settings

## Requirements

- Python 3.8+
- API keys for the LLM providers you want to use
- Dependencies listed in `requirements.txt`
