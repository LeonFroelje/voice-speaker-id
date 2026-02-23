import argparse
import os
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SpeakerSettings(BaseSettings):
    # --- MQTT Connection ---
    mqtt_host: str = Field(
        default="localhost", description="Mosquitto broker IP/Hostname"
    )
    mqtt_port: int = Field(default=1883, description="Mosquitto broker port")
    mqtt_user: Optional[str] = Field(
        default=None, description="Username used to authenticate with mqtt broker"
    )
    mqtt_password: Optional[str] = Field(
        default=None, description="Password used to authenticate with mqtt broker"
    )

    # --- Object Storage (S3 Compatible) ---
    s3_endpoint: str = Field(
        default="http://localhost:3900", description="URL to S3 storage"
    )
    s3_access_key: str = Field(default="your-access-key", description="S3 Access Key")
    s3_secret_key: SecretStr = Field(
        default="your-secret-key", description="S3 Secret Key"
    )
    s3_bucket: str = Field(default="voice-commands", description="S3 Bucket Name")

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
        description="Cosine similarity threshold (0.0 to 1.0)",
    )
    ema_alpha: float = Field(
        default=0.05,
        description="Exponential Moving Average alpha for profile updates",
    )

    # --- System ---
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="SPEAKER_")


def get_settings() -> SpeakerSettings:
    parser = argparse.ArgumentParser(description="Speaker Identification Worker")
    parser.add_argument("--mqtt-host", help="Hostname or IP address")
    parser.add_argument("--mqtt-port", type=int, help="Port of the server")
    parser.add_argument("--mqtt-user")
    parser.add_argument("--mqtt-password")

    parser.add_argument("--s3-endpoint", help="URL to S3 storage")
    parser.add_argument("--s3-access-key", help="S3 Access Key")
    parser.add_argument("--s3-secret-key", help="S3 Secret Key")
    parser.add_argument("--s3-bucket", help="S3 Bucket Name")
    parser.add_argument("--model-path", help="Path to the ONNX model")
    parser.add_argument("--db-path", help="Path to the embeddings JSON file")
    parser.add_argument("--threshold", type=float, help="Cosine similarity threshold")
    parser.add_argument("--ema-alpha", type=float, help="EMA alpha value")
    parser.add_argument("--log-level", help="Logging Level (DEBUG, INFO, ERROR)")

    args, unknown = parser.parse_known_args()
    cli_args = {k.replace("-", "_"): v for k, v in vars(args).items() if v is not None}
    return SpeakerSettings(**cli_args)


settings = get_settings()
