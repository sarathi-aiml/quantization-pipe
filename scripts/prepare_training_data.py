"""Format downloaded datasets for Qwen2.5-VL instruction tuning."""

import json
import logging
import os
import pickle
import random
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
FORMATTED_DIR = PROJECT_ROOT / "datasets" / "formatted"
IMAGES_DIR = FORMATTED_DIR / "images"

RANDOM_SEED = 42

# Diverse medical extraction prompts
MEDICAL_PROMPTS = [
    "Extract all medical terms, diagnoses, and clinical findings from this image.",
    "What diagnoses, conditions, or abnormalities are shown in this image?",
    "List all medications, drug names, and dosages visible in this document.",
    "Identify all clinical measurements, lab values, and vital signs in this image.",
    "Describe the medical findings and observations present in this image.",
    "Extract the key medical information from this document image.",
    "What medical conditions, treatments, or procedures are described in this image?",
    "Summarize the clinical content of this medical document image.",
    "List all anatomical structures, pathological findings, and measurements visible.",
    "Extract patient information, diagnoses, and treatment plans from this document.",
]


def render_text_to_image(text: str, output_path: Path, max_width: int = 80) -> bool:
    """Render text onto a white background image simulating a scanned document."""
    # Wrap text to fit image width
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            wrapped = textwrap.wrap(paragraph, width=max_width)
            lines.extend(wrapped)
        else:
            lines.append("")

    # Limit to reasonable length
    lines = lines[:50]

    width = 800
    height = max(400, min(1200, 60 + len(lines) * 18))
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    y = 30
    for line in lines:
        draw.text((30, y), line, fill=(0, 0, 0), font=font)
        y += 18
        if y > height - 30:
            break

    img.save(output_path, "PNG")
    return True


def process_roco(rng: random.Random) -> list[dict[str, Any]]:
    """Process ROCO / PathVQA dataset (image + text pairs)."""
    pkl_path = RAW_DIR / "roco" / "samples.pkl"
    if not pkl_path.exists():
        logger.warning("ROCO samples not found")
        return []

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    logger.info(f"Processing {len(samples)} ROCO/PathVQA samples...")
    formatted = []
    count = 0

    for i, sample in enumerate(samples):
        # PathVQA format: has 'image', 'question', 'answer'
        image = sample.get("image")
        answer = sample.get("answer", "")
        question = sample.get("question", "")

        if image is None or len(str(answer)) < 20:
            continue

        # Save image
        img_filename = f"roco_{i:05d}.png"
        img_path = IMAGES_DIR / img_filename

        try:
            if isinstance(image, Image.Image):
                image.save(img_path, "PNG")
            else:
                continue
        except Exception:
            continue

        # Use the original question or a random medical prompt
        prompt = question if question else rng.choice(MEDICAL_PROMPTS)
        response = str(answer)

        formatted.append({
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "prompt": prompt,
            "response": response,
            "source": "PathVQA",
        })
        count += 1

    logger.info(f"ROCO/PathVQA: formatted {count} samples")
    return formatted


def process_pubmedvision(rng: random.Random) -> list[dict[str, Any]]:
    """Process PubMedVision dataset (image + text pairs)."""
    pkl_path = RAW_DIR / "pubmedvision" / "samples.pkl"
    if not pkl_path.exists():
        logger.warning("PubMedVision samples not found")
        return []

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    logger.info(f"Processing {len(samples)} PubMedVision samples...")
    formatted = []
    count = 0

    for i, sample in enumerate(samples):
        image = sample.get("image")
        # PubMedVision Alignment VQA format
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not answer or len(str(answer)) < 50:
            continue

        if image is None:
            continue

        img_filename = f"pubmed_{i:05d}.png"
        img_path = IMAGES_DIR / img_filename

        try:
            if isinstance(image, Image.Image):
                image.save(img_path, "PNG")
            else:
                continue
        except Exception:
            continue

        prompt = question if question else rng.choice(MEDICAL_PROMPTS)
        response = str(answer)

        formatted.append({
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "prompt": prompt,
            "response": response,
            "source": "PubMedVision",
        })
        count += 1

    logger.info(f"PubMedVision: formatted {count} samples")
    return formatted


def process_mtsamples(rng: random.Random) -> list[dict[str, Any]]:
    """Process MTSamples dataset (text-only, render to images)."""
    ds_path = RAW_DIR / "mtsamples" / "dataset"
    if not ds_path.exists():
        logger.warning("MTSamples dataset not found")
        return []

    ds = load_from_disk(str(ds_path))
    logger.info(f"Processing {len(ds)} MTSamples...")
    formatted = []
    count = 0

    for i, sample in enumerate(ds):
        # rungalileo/medical_transcription_40 has 'text' and 'label' fields
        text = sample.get("text", "") or sample.get("transcription", "")
        label = sample.get("label", "") or sample.get("medical_specialty", "")

        if not text or len(text) < 50:
            continue

        # Render text to image
        img_filename = f"mtsample_{i:05d}.png"
        img_path = IMAGES_DIR / img_filename

        try:
            render_text_to_image(text[:2000], img_path)
        except Exception:
            continue

        prompt = rng.choice(MEDICAL_PROMPTS)

        # Use the text itself as the response (the model should learn to read it)
        # Truncate very long texts
        response = text[:1500]

        formatted.append({
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "prompt": prompt,
            "response": response,
            "source": "MTSamples",
        })
        count += 1

    logger.info(f"MTSamples: formatted {count} samples")
    return formatted


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # Process all datasets
    all_samples = []
    all_samples.extend(process_roco(rng))
    all_samples.extend(process_pubmedvision(rng))
    all_samples.extend(process_mtsamples(rng))

    logger.info(f"Total formatted samples: {len(all_samples)}")

    # Shuffle with fixed seed
    rng.shuffle(all_samples)

    # Split 80/10/10
    n = len(all_samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_set = all_samples[:train_end]
    val_set = all_samples[train_end:val_end]
    test_set = all_samples[val_end:]

    # Save splits
    for name, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = FORMATTED_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {name}.json: {len(data)} samples")

    # Save summary
    source_counts = {}
    for s in all_samples:
        src = s["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    summary = {
        "total_samples": n,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "source_counts": source_counts,
        "random_seed": RANDOM_SEED,
        "split_ratio": "80/10/10",
    }
    with open(FORMATTED_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset Preparation Summary:")
    print(f"  Total samples: {n}")
    print(f"  Train: {len(train_set)}")
    print(f"  Val: {len(val_set)}")
    print(f"  Test: {len(test_set)}")
    print(f"  Sources: {source_counts}")

    # Spot check 3 random training samples
    print("\nSpot check (3 random training samples):")
    for s in rng.sample(train_set, min(3, len(train_set))):
        img_path = PROJECT_ROOT / s["image_path"]
        print(f"  Image: {s['image_path']} (exists: {img_path.exists()})")
        print(f"  Prompt: {s['prompt'][:80]}...")
        print(f"  Response: {s['response'][:80]}...")
        print()


if __name__ == "__main__":
    main()
