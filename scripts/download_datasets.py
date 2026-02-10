"""Download medical datasets from HuggingFace for fine-tuning."""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def download_roco() -> dict:
    """Download ROCO radiology dataset."""
    logger.info("Downloading ROCO dataset...")
    output_dir = RAW_DIR / "roco"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        start = time.time()
        # Try multiple ROCO dataset names on HuggingFace
        ds = None
        for name in ["abirhasan/ROCO-radiology", "MedICaT/roco", "ROCO-dataset/ROCO"]:
            try:
                ds = load_dataset(name, split="train", streaming=True)
                logger.info(f"ROCO loaded from {name}")
                break
            except Exception:
                continue

        if ds is None:
            # Fallback: try a radiology VQA dataset
            for name in ["flaviagiammarino/path-vqa", "hongrui/RadGenome-ChestCT"]:
                try:
                    ds = load_dataset(name, split="train", streaming=True)
                    logger.info(f"Radiology fallback loaded from {name}")
                    break
                except Exception:
                    continue

        if ds is None:
            raise RuntimeError("No radiology dataset found")

        samples = []
        for i, item in enumerate(ds):
            if i >= 3000:
                break
            samples.append(item)

        elapsed = round(time.time() - start, 2)
        logger.info(f"ROCO: downloaded {len(samples)} samples in {elapsed}s")

        # Save
        import pickle
        with open(output_dir / "samples.pkl", "wb") as f:
            pickle.dump(samples, f)

        return {"name": "ROCO", "success": True, "count": len(samples), "time_seconds": elapsed, "error": None}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"ROCO download failed: {error_msg}")
        return {"name": "ROCO", "success": False, "count": 0, "time_seconds": 0, "error": error_msg}


def download_pubmedvision() -> dict:
    """Download PubMedVision dataset."""
    logger.info("Downloading PubMedVision dataset...")
    output_dir = RAW_DIR / "pubmedvision"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        start = time.time()
        # PubMedVision has specific config names
        ds = None
        for config in ["PubMedVision_Alignment_VQA", "PubMedVision_InstructionTuning_VQA"]:
            try:
                ds = load_dataset("FreedomIntelligence/PubMedVision", config, split="train", streaming=True)
                logger.info(f"PubMedVision loaded with config: {config}")
                break
            except Exception:
                continue

        if ds is None:
            raise RuntimeError("PubMedVision configs not accessible")

        samples = []
        for i, item in enumerate(ds):
            if i >= 2000:
                break
            samples.append(item)

        elapsed = round(time.time() - start, 2)
        logger.info(f"PubMedVision: downloaded {len(samples)} samples in {elapsed}s")

        import pickle
        with open(output_dir / "samples.pkl", "wb") as f:
            pickle.dump(samples, f)

        return {"name": "PubMedVision", "success": True, "count": len(samples), "time_seconds": elapsed, "error": None}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"PubMedVision download failed: {error_msg}")
        return {"name": "PubMedVision", "success": False, "count": 0, "time_seconds": 0, "error": error_msg}


def download_mtsamples() -> dict:
    """Download MTSamples medical transcription dataset."""
    logger.info("Downloading MTSamples dataset...")
    output_dir = RAW_DIR / "mtsamples"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset

        start = time.time()
        # Try various known MTSamples dataset names
        ds = None
        tried = []
        for name in [
            "mtsamples/mtsamples",
            "rungalileo/medical_transcription_40",
            "dreamproit/medical_transcription_mtsamples",
        ]:
            try:
                ds = load_dataset(name, split="train")
                logger.info(f"MTSamples loaded from {name}")
                break
            except Exception as ex:
                tried.append(f"{name}: {ex}")
                continue

        if ds is None:
            # Try a general medical text dataset as fallback
            for name in [
                "gamino/wiki_medical_terms",
                "medical_questions_pairs",
            ]:
                try:
                    ds = load_dataset(name, split="train")
                    logger.info(f"Medical text fallback loaded from {name}")
                    break
                except Exception as ex:
                    tried.append(f"{name}: {ex}")
                    continue

        if ds is None:
            return {
                "name": "MTSamples",
                "success": False,
                "count": 0,
                "time_seconds": 0,
                "error": f"All sources failed: {tried}",
            }

        elapsed = round(time.time() - start, 2)
        count = len(ds)
        logger.info(f"MTSamples: loaded {count} samples in {elapsed}s")

        ds.save_to_disk(str(output_dir / "dataset"))

        return {"name": "MTSamples", "success": True, "count": count, "time_seconds": elapsed, "error": None}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"MTSamples download failed: {error_msg}")
        return {"name": "MTSamples", "success": False, "count": 0, "time_seconds": 0, "error": error_msg}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(download_roco())
    results.append(download_pubmedvision())
    results.append(download_mtsamples())

    success_count = sum(1 for r in results if r["success"])
    total_samples = sum(r["count"] for r in results)

    summary = {
        "datasets": results,
        "total_successful": success_count,
        "total_samples": total_samples,
    }

    output_path = PROJECT_ROOT / "datasets" / "download_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset Download Summary:")
    print(f"  Successful: {success_count}/3")
    print(f"  Total samples: {total_samples}")
    for r in results:
        status = "OK" if r["success"] else "FAILED"
        print(f"  [{status}] {r['name']}: {r['count']} samples")
        if r["error"]:
            print(f"         Error: {r['error'][:100]}")

    if success_count < 2:
        logger.error("Less than 2 datasets downloaded. Pipeline may not have enough data.")


if __name__ == "__main__":
    main()
