"""Benchmark the BNB NF4 quantized Qwen2.5-VL-7B-Instruct on same test images."""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from blackwell_compat import apply_blackwell_patches
apply_blackwell_patches()

from PIL import Image
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "test_images"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def get_gpu_memory_mb() -> dict[str, float]:
    """Get current GPU memory usage in MB."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 2),
    }


def main() -> None:
    # Load the baseline benchmark to get prompts
    with open(BENCHMARKS_DIR / "baseline_fp16.json") as f:
        baseline = json.load(f)

    test_cases = [
        {"image": r["image"], "prompt": r["prompt"]}
        for r in baseline["inference_results"]
    ]

    # Load quantized model
    logger.info("Loading Qwen2.5-VL-7B with BNB NF4 quantization...")
    torch.cuda.reset_peak_memory_stats()
    pre_load_mem = get_gpu_memory_mb()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    total_vram = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_vram * 0.9), "cpu": "32GB"}

    load_start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    load_time = round(time.time() - load_start, 2)

    post_load_mem = get_gpu_memory_mb()
    logger.info(f"Model loaded in {load_time}s, VRAM: {post_load_mem['allocated_mb']} MB")

    # Run inference on same 5 test images with same prompts
    results = []
    latencies = []

    for tc in test_cases:
        image_path = TEST_IMAGES_DIR / tc["image"]
        image = Image.open(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": tc["prompt"]},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        torch.cuda.synchronize()
        latency = round(time.time() - start, 4)
        latencies.append(latency)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        results.append({
            "image": tc["image"],
            "prompt": tc["prompt"],
            "response": response,
            "latency_seconds": latency,
        })
        logger.info(f"  {tc['image']}: {latency}s, response length: {len(response)} chars")

    peak_mem = get_gpu_memory_mb()

    # Compile benchmark
    benchmark = {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "precision": "NF4 (bitsandbytes)",
        "quantization_method": "bitsandbytes_NF4",
        "load_time_seconds": load_time,
        "vram": {
            "pre_load_mb": pre_load_mem,
            "post_load_mb": post_load_mem,
            "peak_mb": peak_mem,
        },
        "inference_results": results,
        "latency_stats": {
            "avg_seconds": round(sum(latencies) / len(latencies), 4),
            "min_seconds": round(min(latencies), 4),
            "max_seconds": round(max(latencies), 4),
            "p50_seconds": round(sorted(latencies)[len(latencies) // 2], 4),
        },
    }

    output_path = BENCHMARKS_DIR / "quantized_int4.json"
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    logger.info(f"Quantized benchmark saved to {output_path}")

    # Generate comparison table
    fp16_vram = baseline["vram"]["post_load_mb"]["allocated_mb"]
    nf4_vram = post_load_mem["allocated_mb"]
    vram_reduction = round((1 - nf4_vram / fp16_vram) * 100, 1)

    fp16_lat = baseline["latency_stats"]["avg_seconds"]
    nf4_lat = benchmark["latency_stats"]["avg_seconds"]
    lat_change = round((nf4_lat / fp16_lat - 1) * 100, 1)

    print("\n" + "=" * 70)
    print("FP16 vs NF4 QUANTIZATION COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'FP16':>15} {'NF4':>15} {'Change':>10}")
    print("-" * 70)
    print(f"{'VRAM (post-load) MB':<30} {fp16_vram:>15.1f} {nf4_vram:>15.1f} {vram_reduction:>9.1f}%")
    print(f"{'VRAM (peak) MB':<30} {baseline['vram']['peak_mb']['max_allocated_mb']:>15.1f} {peak_mem['max_allocated_mb']:>15.1f}")
    print(f"{'Load Time (s)':<30} {baseline['load_time_seconds']:>15.2f} {load_time:>15.2f}")
    print(f"{'Avg Latency (s)':<30} {fp16_lat:>15.4f} {nf4_lat:>15.4f} {lat_change:>+9.1f}%")
    print(f"{'Min Latency (s)':<30} {baseline['latency_stats']['min_seconds']:>15.4f} {benchmark['latency_stats']['min_seconds']:>15.4f}")
    print(f"{'Max Latency (s)':<30} {baseline['latency_stats']['max_seconds']:>15.4f} {benchmark['latency_stats']['max_seconds']:>15.4f}")
    print("-" * 70)
    print("\nResponse coherence check:")
    for i, (fp16_r, nf4_r) in enumerate(zip(baseline["inference_results"], results)):
        fp16_preview = fp16_r["response"][:60].replace("\n", " ")
        nf4_preview = nf4_r["response"][:60].replace("\n", " ")
        print(f"  Image {i+1} ({fp16_r['image']}):")
        print(f"    FP16: {fp16_preview}...")
        print(f"    NF4:  {nf4_preview}...")
    print("=" * 70)

    # Clean up
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU cache cleared.")


if __name__ == "__main__":
    main()
