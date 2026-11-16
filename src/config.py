"""
Configuration settings for Face Recognition service and CLI.
Loads settings from environment variables and .env file with safe defaults.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def load_env_file() -> None:
    """
    Load environment variables from .env file into os.environ if not already set.
    Checks the project root and current working directory without external dependencies.
    """
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / ".env",
        Path.cwd() / ".env",
    ]
    seen_paths = set()
    for env_path in candidate_paths:
        try:
            resolved = env_path.resolve()
        except Exception:
            resolved = env_path
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (
                            val.startswith("'") and val.endswith("'")
                        ):
                            val = val[1:-1]
                        if key not in os.environ:
                            os.environ[key] = val
                break
            except Exception:
                pass


# Load .env before initializing settings
load_env_file()


@dataclass(frozen=True)
class Settings:
    # Hugging Face Model Hub Settings
    hf_repo_id: str = os.getenv("HF_REPO_ID", "AkihikoSaburo/methics")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    models_config_file: str = os.getenv("MODELS_CONFIG_FILE", "models_config.json")

    # Default Models (configured via .env or environment variables)
    det_model_path: str = os.getenv(
        "DET_MODEL_PATH", "face_detection/scrfd_2.5g_bnkps.onnx"
    )
    rec_model_path: str = os.getenv(
        "REC_MODEL_PATH", "face_recognition/arcface_r50_fp16.onnx"
    )

    # Detection & Matching Thresholds
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.50"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.40"))
    sim_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.45"))

    # Server Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Security
    api_token: Optional[str] = os.getenv("API_TOKEN")

    # Image Processing Defaults
    aligned_face_size: int = int(os.getenv("ALIGNED_FACE_SIZE", "112"))
    input_tensor_size: int = int(os.getenv("INPUT_TENSOR_SIZE", "640"))


settings = Settings()
