# Layered Reasoning Chatbot

A proof-of-concept implementation of a chatbot with enhanced reasoning capabilities through a layered architecture.

## Features

- **Intent Recognition**: Uses advanced NLU to understand user meaning beyond literal words
- **Strategy Formation**: Employs Bayesian decision-making to choose optimal response strategies
- **Response Generation**: Generates contextually appropriate responses guided by chosen strategies
- **Memory Management**: Maintains conversation history and episodic memory
- **Explainable AI**: Can provide reasoning behind its responses
- **Adaptive Learning**: Improves response strategies based on feedback

## Architecture

The bot uses a three-layer architecture:

1. **Intent Recognition Layer**

   - Parses user input
   - Infers deeper meaning and context
   - Analyzes emotional tone and implicit assumptions

2. **Strategy Formation Layer**

   - Decides response approach using Bayesian decision-making
   - Considers multiple strategy options:
     - Factual explanation
     - Clarifying questions
     - Examples and analogies
     - Question reframing

3. **Response Generation Layer**
   - Generates responses using selected strategy
   - Maintains consistent tone and style
   - Incorporates conversation context

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up OpenAI API key:

   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

3. Configure the bot in `config.yaml`

## Usage

Run the interactive demo:

```bash
python layered_reasoning_bot.py
```

Example interaction:

```
You: What is artificial intelligence?
Bot: [Response with layered reasoning]

You: How does it affect society?
Bot: [Contextual response considering previous interaction]
```

## Configuration

The bot's behavior can be customized through `config.yaml`:

- Model parameters (temperature, max tokens)
- Strategy formation settings
- Memory configuration
- Feature toggles

## Development

The codebase follows these principles:

- Type hints for better code understanding
- Comprehensive logging
- Error handling with retries
- Modular architecture for easy extension

## Requirements

- Python 3.8+
- OpenAI API access
- Dependencies listed in requirements.txt
