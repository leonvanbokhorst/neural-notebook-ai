# Voice Blending POC

A proof of concept demonstrating voice blending capabilities using the Kokoro TTS model. This POC explores different voice interpolation techniques to create unique synthetic voices by blending multiple voice embeddings.

## Overview

This POC implements and demonstrates:

- Loading and managing voice embeddings from the Kokoro TTS model
- Linear interpolation between voice embeddings
- Spherical linear interpolation (SLERP) for more natural voice blending
- Generation of speech samples using blended voices
- Configurable voice blending parameters and methods

## Technical Details

### Voice Blending Techniques

1. **Linear Interpolation**

   - Simple weighted averaging between voice embeddings
   - Formula: `blended_voice = (1 - t) * voice_a + t * voice_b`
   - Good for small differences between voices

2. **Spherical Linear Interpolation (SLERP)**
   - Interpolation along the geodesic of the voice embedding manifold
   - Preserves the natural structure of the voice space
   - Better for voices with significant differences
   - Formula:
     ```python
     theta = acos(dot(voice_a, voice_b))
     blended_voice = sin((1-t)*theta)/sin(theta) * voice_a + sin(t*theta)/sin(theta) * voice_b
     ```

### Architecture

- Event-driven processing for voice generation
- Asynchronous audio processing
- Configurable voice blending parameters
- Extensible blending method framework

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- espeak-ng (for phonemization)
- git-lfs (for model download)
- Dependencies listed in `requirements.txt`

## Installation

1. Install system dependencies:

```bash
# On Ubuntu/Debian
sudo apt-get install espeak-ng git-lfs

# On macOS
brew install espeak-ng git-lfs
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

The Kokoro model will be automatically downloaded when you first run the POC.

## Usage

1. Configure the POC in `config.yaml`:

   - Set model path and parameters
   - Configure voice directories
   - Adjust blending parameters

2. Place your voice embedding files (`.pt` format) in the `voices` directory.

3. Run the demonstration:

```bash
python voice_blending.py
```

The script will:

- Download the Kokoro model if not present
- Generate samples of original voices
- Create voice blends using linear interpolation
- Create voice blends using SLERP
- Save all generated audio files in the output directory

## Project Structure

```
voice_blending/
├── README.md              # This file
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
├── voice_blending.py     # Main implementation
├── models/              # Downloaded model files
│   └── Kokoro-82M/     # Kokoro model directory
├── voices/              # Voice embedding files
│   ├── bf_emma.pt
│   ├── bf_isabella.pt
│   └── bm_lewis.pt
└── output/              # Generated audio files
```

## Development

### Code Quality

- Follow PEP 8 style guide
- Use type hints (PEP 484)
- Maintain test coverage (pytest)
- Run linting (pylint)
- Check types (mypy)

### Testing

Run tests with:

```bash
pytest tests/ --cov=voice_blending
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) for the base TTS model
- Voice embeddings from the Kokoro model repository
