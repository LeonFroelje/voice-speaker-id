import argparse
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SpeakerSettings(BaseSettings):
    # --- Server Settings ---
    host: str = Field(
        default="127.0.0.1",
        description="The Hostname or IP address to bind the server to",
    )
    port: int = Field(default=8001, description="The port of the FastAPI server")

    # --- Model & Data Settings ---
    model_path: str = Field(
        default="/var/lib/speaker-api/cam++.onnx",
        description="Path to the ONNX speaker embedding model",
    )
    db_path: str = Field(
        default="/var/lib/speaker-api/embeddings.json",
        description="Path to the JSON database storing speaker embeddings",
    )

    # --- Algorithm Settings ---
    threshold: float = Field(
        default=0.5,
        description="Cosine similarity threshold (0.0 to 1.0) for a positive identification",
    )
    ema_alpha: float = Field(
        default=0.05,
        description="Exponential Moving Average alpha for profile updates (0.0 to 1.0)",
    )

    # Pydantic Config: Tells it to read from environment variables with this prefix
    model_config = SettingsConfigDict(env_prefix="SPEAKER_")


def get_settings() -> SpeakerSettings:
    """
    Parses CLI arguments first, then initializes Settings.
    Precedence: CLI Args > Environment Vars > .env file > Defaults
    """
    parser = argparse.ArgumentParser(description="Speaker Identification API")

    # Add arguments for every field you want controllable via CLI
    parser.add_argument("--host", help="Hostname or IP address")
    parser.add_argument("--port", type=int, help="Port of the server")
    parser.add_argument("--model-path", help="Path to the ONNX model")
    parser.add_argument("--db-path", help="Path to the embeddings JSON file")
    parser.add_argument("--threshold", type=float, help="Cosine similarity threshold")
    parser.add_argument(
        "--ema-alpha", type=float, help="EMA alpha value for profile updates"
    )

    args, unknown = parser.parse_known_args()

    # Create a dictionary of only the arguments that were actually provided via CLI
    # We replace hyphens with underscores to match the Pydantic field names
    cli_args = {k.replace("-", "_"): v for k, v in vars(args).items() if v is not None}

    # Initialize Settings
    return SpeakerSettings(**cli_args)


# Create a global instance
settings = get_settings()
