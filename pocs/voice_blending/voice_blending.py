"""
Voice Blending POC using Kokoro TTS
----------------------------------

Demonstrates voice blending capabilities using Kokoro TTS model.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import yaml
from scipy.io.wavfile import write
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model URLs and files
MODEL_FILES = {
    "kokoro-v0_19.pth": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.pth",
    "models.py": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/models.py",
    "kokoro.py": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/kokoro.py",
    "plbert.py": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/plbert.py",
    "istftnet.py": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/istftnet.py",
}

VOICE_FILES = [
    "bf_emma.pt",
    "bf_isabella.pt",
    "bm_lewis.pt",
]


def check_command_exists(command: str) -> bool:
    """Check if a command exists in the system PATH."""
    if platform.system() == "Windows":
        cmd = f"where {command}"
    else:
        cmd = f"which {command}"

    try:
        subprocess.run(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def download_file(url: str, dest_path: Path, desc: Optional[str] = None) -> None:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        desc = desc or dest_path.name
        with open(dest_path, "wb") as file, tqdm(
            desc=desc,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise


def get_device(device: Optional[str] = None) -> str:
    """Get the appropriate device for PyTorch computation.

    Args:
        device: Optional device specification. If None, will auto-detect.

    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu')
    """
    if device and device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ConfigurationParams:
    """Configuration parameters for the Voice Blending POC."""

    model_path: str
    voices_dir: str
    output_dir: str
    sample_rate: int = 24000
    device: str = get_device()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigurationParams":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ConfigurationParams":
        """Load configuration from YAML file."""
        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)

            # Handle device configuration
            device = get_device(config_dict["model"]["device"])
            logger.info(f"Using device: {device}")

            return cls(
                model_path=config_dict["model"]["path"],
                voices_dir=config_dict["voices"]["directory"],
                output_dir=config_dict["output"]["directory"],
                sample_rate=config_dict["model"]["sample_rate"],
                device=device,
            )
        except (yaml.YAMLError, KeyError) as e:
            logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            raise


class VoiceBlender:
    """Handles voice blending operations."""

    def __init__(self, device: str):
        self.device = torch.device(device)

    def linear_interpolation(
        self, voice_a: torch.Tensor, voice_b: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Linear interpolation between two voice embeddings."""
        return (1 - t) * voice_a + t * voice_b

    def slerp(
        self, voice_a: torch.Tensor, voice_b: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Spherical linear interpolation between two voice embeddings."""
        dot = torch.sum(voice_a * voice_b, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        if sin_theta.max().item() < 1e-6:
            return self.linear_interpolation(voice_a, voice_b, t)

        sin_t_theta = torch.sin(t * theta)
        sin_one_minus_t_theta = torch.sin((1 - t) * theta)

        return (sin_one_minus_t_theta / sin_theta) * voice_a + (
            sin_t_theta / sin_theta
        ) * voice_b


class POCImplementation:
    """Main implementation class for the Voice Blending POC."""

    def __init__(self, config: ConfigurationParams):
        self.config = config
        self.blender = VoiceBlender(config.device)
        self.model = None
        self.voices = {}
        self.base_dir = Path(__file__).parent.absolute()
        self._setup()

    def _check_espeak(self) -> None:
        """Check if espeak-ng is installed."""
        if not check_command_exists("espeak-ng"):
            # Try alternative command on macOS
            if platform.system() == "Darwin" and check_command_exists("espeak"):
                return

            # Not found, provide installation instructions
            os_specific = ""
            if platform.system() == "Darwin":
                os_specific = (
                    "On macOS:\n"
                    "1. Install Homebrew if not installed:\n"
                    '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
                    "2. Install espeak-ng:\n"
                    "   brew install espeak-ng"
                )
            elif platform.system() == "Linux":
                os_specific = (
                    "On Ubuntu/Debian:\n"
                    "   sudo apt-get install espeak-ng\n"
                    "On Fedora:\n"
                    "   sudo dnf install espeak-ng"
                )
            elif platform.system() == "Windows":
                os_specific = (
                    "On Windows:\n"
                    "1. Download the installer from: https://github.com/espeak-ng/espeak-ng/releases\n"
                    "2. Run the installer and follow the instructions"
                )

            raise RuntimeError(
                "espeak-ng is required but not installed.\n\n"
                f"Installation instructions:\n{os_specific}"
            )

    def _download_model(self) -> None:
        """Download the Kokoro model files."""
        model_dir = self.base_dir / "models" / "Kokoro-82M"
        if not model_dir.exists():
            logger.info("Downloading Kokoro model files...")
            os.makedirs(model_dir, exist_ok=True)

            # Download model files
            for filename, url in MODEL_FILES.items():
                dest_path = model_dir / filename
                if not dest_path.exists():
                    try:
                        download_file(url, dest_path)
                    except Exception as e:
                        logger.error(f"Failed to download {filename}: {e}")
                        if dest_path.exists():
                            dest_path.unlink()
                        raise

            logger.info("Model files downloaded successfully")

    def _setup_voices(self) -> None:
        """Set up voice embedding files."""
        voices_dir = self.base_dir / self.config.voices_dir
        os.makedirs(voices_dir, exist_ok=True)

        # Download voice files
        for voice_file in VOICE_FILES:
            dest_path = voices_dir / voice_file
            if not dest_path.exists():
                url = f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice_file}"
                try:
                    download_file(url, dest_path, f"Downloading {voice_file}")
                except Exception as e:
                    logger.error(f"Failed to download voice file {voice_file}: {e}")
                    if dest_path.exists():
                        dest_path.unlink()
                    raise

    def _setup(self) -> None:
        """Initialize the TTS model and load voice embeddings."""
        logger.info("Setting up Voice Blending POC environment")

        try:
            # Check for espeak-ng
            self._check_espeak()

            # Download model and voice files
            self._download_model()
            self._setup_voices()

            # Add model directory to Python path
            model_dir = self.base_dir / "models" / "Kokoro-82M"
            if str(model_dir) not in sys.path:
                sys.path.insert(0, str(model_dir))

            from kokoro import generate
            from models import build_model

            model_path = model_dir / self.config.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = build_model(str(model_path), self.config.device)
            self.generate = generate

            # Create output directory
            Path(self.base_dir / self.config.output_dir).mkdir(
                parents=True, exist_ok=True
            )

        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            raise

    def load_voice(self, voice_name: str) -> torch.Tensor:
        """Load a voice embedding from file."""
        if voice_name not in self.voices:
            voice_path = self.base_dir / self.config.voices_dir / f"{voice_name}.pt"
            if not voice_path.exists():
                raise FileNotFoundError(f"Voice file not found: {voice_path}")

            self.voices[voice_name] = torch.load(voice_path, weights_only=True).to(
                self.config.device
            )
        return self.voices[voice_name]

    def generate_speech(
        self, text: str, voice_embedding: torch.Tensor, output_file: str
    ) -> None:
        """Generate speech using the provided voice embedding."""
        audio, phonemes = self.generate(self.model, text, voice_embedding, lang="b")

        # Convert to numpy array and normalize
        audio_np = np.array(audio, dtype=np.float32)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Save to WAV file
        output_path = self.base_dir / self.config.output_dir / output_file
        write(str(output_path), self.config.sample_rate, audio_int16)
        logger.info(f"Generated speech saved to: {output_path}")
        logger.debug(f"Phonemes used: {phonemes}")

    def demonstrate_concept(self) -> List[str]:
        """Demonstrate voice blending capabilities."""
        logger.info("Running voice blending demonstration")

        # Test text
        text = "Voice blending allows us to create unique and expressive synthetic voices. Wheeeee"

        # Load voice embeddings
        voice_emma = self.load_voice("bf_emma")
        voice_isabella = self.load_voice("bf_isabella")
        voice_lewis = self.load_voice("bm_lewis")

        generated_files = []

        # Generate original voices
        for name, voice in [
            ("emma", voice_emma),
            ("isabella", voice_isabella),
            ("lewis", voice_lewis),
        ]:
            output_file = f"original_{name}.wav"
            self.generate_speech(text, voice, output_file)
            generated_files.append(output_file)

        # Demonstrate linear interpolation
        for t in [0.3, 0.5, 0.7]:
            blended_voice = self.blender.linear_interpolation(
                voice_emma, voice_lewis, t
            )
            output_file = f"linear_blend_{t:.1f}.wav"
            self.generate_speech(text, blended_voice, output_file)
            generated_files.append(output_file)

        # Demonstrate SLERP
        for t in [0.3, 0.5, 0.7]:
            blended_voice = self.blender.slerp(voice_emma, voice_lewis, t)
            output_file = f"slerp_blend_{t:.1f}.wav"
            self.generate_speech(text, blended_voice, output_file)
            generated_files.append(output_file)

        return generated_files

    def cleanup(self) -> None:
        """Cleanup any resources."""
        logger.info("Cleaning up resources")
        self.model = None
        self.voices.clear()
        torch.cuda.empty_cache()


def main():
    """Main entry point for the POC."""
    try:
        # Load configuration from YAML
        config_path = Path(__file__).parent / "config.yaml"
        config = ConfigurationParams.from_yaml(str(config_path))

        # Initialize and run demonstration
        poc = POCImplementation(config)
        generated_files = poc.demonstrate_concept()

        # Log results
        logger.info("Demonstration completed successfully")
        logger.info("Generated files:")
        for file in generated_files:
            logger.info(f"- {file}")

    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise
    finally:
        if "poc" in locals():
            poc.cleanup()


if __name__ == "__main__":
    main()
