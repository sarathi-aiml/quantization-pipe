"""Quantize Qwen2.5-VL-7B-Instruct using AWQ (preferred) or BNB NF4 (fallback)."""

import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from blackwell_compat import apply_blackwell_patches
apply_blackwell_patches()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
AWQ_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-awq-int4"
BNB_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-bnb-nf4"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def try_awq_quantization() -> dict:
    """Attempt AWQ INT4 quantization. Returns info dict."""
    logger.info("Attempting AWQ INT4 quantization...")
    try:
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_pretrained(str(MODEL_DIR))
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
        }
        AWQ_DIR.mkdir(parents=True, exist_ok=True)
        start = time.time()
        model.quantize(calib_data="pileval", quant_config=quant_config)
        model.save_quantized(str(AWQ_DIR))
        elapsed = round(time.time() - start, 2)

        return {
            "method": "AWQ_INT4",
            "success": True,
            "time_seconds": elapsed,
            "output_dir": str(AWQ_DIR),
            "error": None,
        }
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"AWQ quantization failed: {error_msg}")
        return {
            "method": "AWQ_INT4",
            "success": False,
            "time_seconds": 0,
            "output_dir": None,
            "error": error_msg,
        }


def setup_bnb_nf4() -> dict:
    """Configure BNB NF4 quantization (loaded on-the-fly from base model)."""
    logger.info("Setting up bitsandbytes NF4 quantization config...")
    from transformers import BitsAndBytesConfig

    start = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Save config to disk so other scripts know how to load it
    BNB_DIR.mkdir(parents=True, exist_ok=True)
    config_data = {
        "method": "bitsandbytes_NF4",
        "base_model_dir": str(MODEL_DIR),
        "config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "notes": "BNB NF4 quantization is applied on-the-fly during model loading. "
                 "No separate quantized model files are saved. Load the base model "
                 "with this BitsAndBytesConfig to get the quantized version.",
    }
    config_path = BNB_DIR / "quantization_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    elapsed = round(time.time() - start, 2)
    logger.info(f"BNB NF4 config saved to {config_path}")

    # Verify we can load with this config
    logger.info("Verifying BNB NF4 model loads correctly...")
    from transformers import Qwen2_5_VLForConditionalGeneration

    total_vram = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_vram * 0.9), "cpu": "32GB"}

    load_start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )
    load_time = round(time.time() - load_start, 2)

    vram_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
    logger.info(f"BNB NF4 model loaded in {load_time}s, VRAM: {vram_mb} MB")

    # Calculate expected VRAM reduction
    # FP16 = 2 bytes/param, NF4 = 0.5 bytes/param (4-bit) + overhead
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "bitsandbytes_NF4",
        "success": True,
        "time_seconds": elapsed,
        "load_time_seconds": load_time,
        "output_dir": str(BNB_DIR),
        "vram_mb": vram_mb,
        "total_params": total_params,
        "error": None,
    }


def main() -> None:
    results = {}

    # Try AWQ first
    awq_result = try_awq_quantization()
    results["awq_attempt"] = awq_result

    if awq_result["success"]:
        results["active_method"] = "AWQ_INT4"
        logger.info("AWQ quantization succeeded!")
    else:
        # Fall back to BNB NF4
        logger.info("Falling back to bitsandbytes NF4 quantization...")
        bnb_result = setup_bnb_nf4()
        results["bnb_setup"] = bnb_result
        results["active_method"] = "bitsandbytes_NF4"

        if bnb_result["success"]:
            logger.info("BNB NF4 quantization setup succeeded!")
        else:
            logger.error("Both AWQ and BNB NF4 failed!")
            sys.exit(1)

    # Save quantization info
    output_path = BENCHMARKS_DIR / "quantization_info.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Quantization info saved to {output_path}")


if __name__ == "__main__":
    main()
