import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent.parent

with open(_ROOT / "configs" / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CLASSES = CONFIG["classes"]["names"]
NUM_CLASSES = CONFIG["classes"]["num_classes"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}
REVERSE_CLASS_MAP = {idx: name for name, idx in CLASS_MAP.items()}

SPRAY_PROFILES = CONFIG["spray"]

MODEL_CFG = CONFIG["model"]
TRAIN_CFG = CONFIG["training"]
AUG_CFG = CONFIG["augmentation"]
INFERENCE_CFG = CONFIG["inference"]
DRONE_CFG = CONFIG["drone"]
CAMERA_CFG = CONFIG["camera"]


def get_project_root() -> Path:
    return _ROOT


def get_device():
    import torch
    device_str = INFERENCE_CFG.get("device", "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def setup_logging(name: str, level: str = None) -> logging.Logger:
    if level is None:
        level = CONFIG["logging"].get("level", "INFO")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level))
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        log_dir = _ROOT / CONFIG["logging"]["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setLevel(getattr(logging, level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def ensure_dirs():
    for key, rel_path in CONFIG["paths"].items():
        full_path = _ROOT / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
