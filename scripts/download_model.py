"""Download Qwen2.5-VL-7B-Instruct from HuggingFace."""

import json
import logging
import time
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LOCAL_DIR = Path(__file__).parent.parent / "models" / "qwen25-vl-7b-base"
BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"


def get_dir_size_gb(path: Path) -> float:
    """Calculate total size of a directory in GB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024**3), 2)


def main() -> None:
    logger.info(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        ignore_patterns=["*.md", "LICENSE*", ".gitattributes"],
    )
    download_time = round(time.time() - start_time, 2)

    model_size_gb = get_dir_size_gb(LOCAL_DIR)
    logger.info(f"Download complete in {download_time}s")
    logger.info(f"Model size on disk: {model_size_gb} GB")

    # Save download info
    download_info = {
        "model_id": MODEL_ID,
        "local_dir": str(LOCAL_DIR),
        "model_size_gb": model_size_gb,
        "download_time_seconds": download_time,
    }
    info_path = BENCHMARKS_DIR / "download_info.json"
    with open(info_path, "w") as f:
        json.dump(download_info, f, indent=2)
    logger.info(f"Download info saved to {info_path}")


if __name__ == "__main__":
    main()
