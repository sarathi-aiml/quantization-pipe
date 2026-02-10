"""FastAPI server for the medical vision pipeline.

Serves the fine-tuned Qwen2.5-VL-7B model (base + LoRA, NF4 quantized) as a
REST API for medical document extraction and analysis.  The model is loaded once
at startup using a lifespan context manager and shared across all requests.

Endpoints
---------
POST /extract  -- Extract medical terms from an uploaded image.
POST /analyze  -- Run custom or default analysis on an uploaded image.
GET  /health   -- Health check with GPU / model status.
GET  /benchmark -- Return all benchmark JSON results.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Blackwell compatibility -- must be imported before any heavy CUDA work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from blackwell_compat import apply_blackwell_patches  # noqa: E402

apply_blackwell_patches()

from transformers import (  # noqa: E402
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("medical_vision_api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
LORA_ADAPTER_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-medical-lora"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata about the model that served the request."""
    model_name: str = "Qwen2.5-VL-7B-Instruct"
    precision: str = "NF4 (bitsandbytes)"
    lora_adapter: str = "qwen25-vl-7b-medical-lora"
    device: str = ""


class ExtractionResponse(BaseModel):
    """Response schema for /extract."""
    extracted_content: str
    inference_latency_seconds: float
    model_info: ModelInfo
    timestamp: str


class AnalysisResponse(BaseModel):
    """Response schema for /analyze."""
    analysis: str
    prompt_used: str
    inference_latency_seconds: float
    model_info: ModelInfo
    timestamp: str


class HealthResponse(BaseModel):
    """Response schema for /health."""
    model_loaded: bool
    gpu_available: bool
    gpu_name: str
    current_vram_usage_mb: float
    total_vram_mb: float
    timestamp: str


class BenchmarkResponse(BaseModel):
    """Response schema for /benchmark."""
    benchmarks: dict[str, Any]
    files_loaded: list[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Generic error body."""
    detail: str
    timestamp: str


# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------

class _ModelState:
    """Mutable singleton that holds the loaded model and processor."""

    def __init__(self) -> None:
        self.model: Qwen2_5_VLForConditionalGeneration | None = None
        self.processor: AutoProcessor | None = None
        self.device: str = "cpu"
        self.loaded: bool = False


_state = _ModelState()

# ---------------------------------------------------------------------------
# Default extraction prompt
# ---------------------------------------------------------------------------
DEFAULT_EXTRACTION_PROMPT = (
    "You are a medical document analysis expert. Carefully examine this medical "
    "document image and extract ALL of the following information:\n"
    "1. Medical diagnoses and conditions\n"
    "2. Medications with dosages and administration routes\n"
    "3. Laboratory values with units and reference ranges\n"
    "4. Vital signs with measurements\n"
    "5. Clinical procedures or recommendations\n\n"
    "Present the extracted information in a clear, structured format using "
    "headings and bullet points. Include all numerical values exactly as shown."
)

# ---------------------------------------------------------------------------
# Model loading / unloading helpers
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """Load the fine-tuned model (base + LoRA merged) with NF4 quantization."""
    logger.info("Loading base model with BNB NF4 quantization ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    total_vram = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_vram * 0.80), "cpu": "32GB"}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )

    # Load LoRA adapter if it exists
    adapter_config_path = LORA_ADAPTER_DIR / "adapter_config.json"
    if adapter_config_path.exists():
        logger.info("Loading LoRA adapter from %s ...", LORA_ADAPTER_DIR)
        model = PeftModel.from_pretrained(model, str(LORA_ADAPTER_DIR))
        try:
            model = model.merge_and_unload()
            logger.info("LoRA adapter merged and unloaded successfully.")
        except Exception as exc:
            logger.warning(
                "merge_and_unload() failed (%s); serving with adapter attached.",
                exc,
            )
    else:
        logger.warning(
            "No LoRA adapter found at %s -- serving base model only.",
            LORA_ADAPTER_DIR,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(str(BASE_MODEL_DIR))

    _state.model = model
    _state.processor = processor
    _state.device = str(next(model.parameters()).device)
    _state.loaded = True

    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    logger.info(
        "Model ready on %s  |  VRAM allocated: %.1f MB",
        _state.device,
        vram_mb,
    )


def _unload_model() -> None:
    """Release model memory."""
    if _state.model is not None:
        del _state.model
    if _state.processor is not None:
        del _state.processor
    _state.model = None
    _state.processor = None
    _state.loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded and GPU cache cleared.")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(image: Image.Image, prompt: str) -> tuple[str, float]:
    """Run a single inference pass and return (response_text, latency_seconds).

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet.
    """
    if not _state.loaded or _state.model is None or _state.processor is None:
        raise RuntimeError("Model is not loaded.")

    model = _state.model
    processor = _state.processor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    torch.cuda.synchronize()
    latency = round(time.perf_counter() - t0, 4)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text: str = processor.decode(generated_ids, skip_special_tokens=True)

    return response_text, latency


def _build_model_info() -> ModelInfo:
    return ModelInfo(
        model_name="Qwen2.5-VL-7B-Instruct",
        precision="NF4 (bitsandbytes)",
        lora_adapter="qwen25-vl-7b-medical-lora",
        device=_state.device,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; unload on shutdown."""
    try:
        _load_model()
    except Exception:
        logger.exception("Failed to load model during startup.")
    yield
    _unload_model()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Vision API",
    description=(
        "REST API serving Qwen2.5-VL-7B-Instruct fine-tuned with LoRA for "
        "medical document understanding.  NF4 quantized via bitsandbytes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: validate image upload
# ---------------------------------------------------------------------------

async def _read_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file into a PIL Image.

    Raises HTTPException(400) when the file is not a valid image.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {exc}",
        )
    return image


def _require_model() -> None:
    """Raise 503 if the model is not loaded."""
    if not _state.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please wait for startup to complete.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Extract medical information from a document image",
)
async def extract(file: UploadFile = File(...)) -> ExtractionResponse:
    """Accept an uploaded medical document image and return structured
    extractions of diagnoses, medications, lab values, and vital signs."""
    _require_model()
    request_start = time.perf_counter()
    image = await _read_image(file)

    logger.info(
        "POST /extract  |  filename=%s  size=%dx%d",
        file.filename,
        image.width,
        image.height,
    )

    try:
        response_text, latency = _run_inference(image, DEFAULT_EXTRACTION_PROMPT)
    except Exception as exc:
        logger.exception("Inference failed on /extract")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    total_time = round(time.perf_counter() - request_start, 4)
    logger.info(
        "POST /extract  |  inference=%.2fs  total=%.2fs  response_len=%d",
        latency,
        total_time,
        len(response_text),
    )

    return ExtractionResponse(
        extracted_content=response_text,
        inference_latency_seconds=latency,
        model_info=_build_model_info(),
        timestamp=_now_iso(),
    )


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Analyze a medical image with an optional custom prompt",
)
async def analyze(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
) -> AnalysisResponse:
    """Accept an uploaded image and an optional custom prompt.  If no prompt is
    provided, a default medical analysis prompt is used."""
    _require_model()
    request_start = time.perf_counter()
    image = await _read_image(file)

    effective_prompt = prompt if prompt else DEFAULT_EXTRACTION_PROMPT

    logger.info(
        "POST /analyze  |  filename=%s  size=%dx%d  custom_prompt=%s",
        file.filename,
        image.width,
        image.height,
        bool(prompt),
    )

    try:
        response_text, latency = _run_inference(image, effective_prompt)
    except Exception as exc:
        logger.exception("Inference failed on /analyze")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    total_time = round(time.perf_counter() - request_start, 4)
    logger.info(
        "POST /analyze  |  inference=%.2fs  total=%.2fs  response_len=%d",
        latency,
        total_time,
        len(response_text),
    )

    return AnalysisResponse(
        analysis=response_text,
        prompt_used=effective_prompt,
        inference_latency_seconds=latency,
        model_info=_build_model_info(),
        timestamp=_now_iso(),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health and GPU status",
)
async def health() -> HealthResponse:
    """Return model readiness, GPU availability, and VRAM usage."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    vram_used_mb = round(
        torch.cuda.memory_allocated(0) / (1024 ** 2), 2
    ) if gpu_available else 0.0
    total_vram_mb = round(
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2
    ) if gpu_available else 0.0

    return HealthResponse(
        model_loaded=_state.loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        current_vram_usage_mb=vram_used_mb,
        total_vram_mb=total_vram_mb,
        timestamp=_now_iso(),
    )


@app.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Return all benchmark results",
)
async def benchmark() -> BenchmarkResponse:
    """Read every JSON file in the benchmarks directory and return the combined
    data in a single response."""
    benchmarks: dict[str, Any] = {}
    files_loaded: list[str] = []

    if BENCHMARKS_DIR.is_dir():
        for json_path in sorted(BENCHMARKS_DIR.glob("*.json")):
            try:
                with open(json_path) as fh:
                    benchmarks[json_path.stem] = json.load(fh)
                files_loaded.append(json_path.name)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", json_path, exc)
                benchmarks[json_path.stem] = {"error": str(exc)}
                files_loaded.append(f"{json_path.name} (error)")

    return BenchmarkResponse(
        benchmarks=benchmarks,
        files_loaded=files_loaded,
        timestamp=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the server via uvicorn."""
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,  # single worker -- model lives in-process
    )


if __name__ == "__main__":
    main()
