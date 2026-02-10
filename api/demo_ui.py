"""Gradio demo UI for the Medical Vision API.

Provides a three-tab interface that communicates with the FastAPI server
(default: http://localhost:8000):

* Tab 1 -- Medical Term Extraction: upload an image, extract medical info.
* Tab 2 -- Custom Analysis: upload an image + provide a free-form prompt.
* Tab 3 -- Benchmarks: load and display all pipeline benchmark data.

Launch
------
    python -m api.demo_ui            # from the project root
    python api/demo_ui.py            # from the project root

The Gradio server binds to 0.0.0.0:7860 by default.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("medical_vision_demo")

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def _api_url(path: str) -> str:
    """Build full API URL from a relative path."""
    return f"{API_BASE_URL}{path}"


def _check_api() -> str:
    """Ping the /health endpoint and return a status string."""
    try:
        resp = requests.get(_api_url("/health"), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        model_status = "loaded" if data.get("model_loaded") else "NOT loaded"
        gpu = data.get("gpu_name", "N/A")
        vram = data.get("current_vram_usage_mb", 0)
        total = data.get("total_vram_mb", 0)
        return (
            f"API reachable  |  Model: {model_status}  |  "
            f"GPU: {gpu}  |  VRAM: {vram:.0f} / {total:.0f} MB"
        )
    except requests.ConnectionError:
        return "ERROR: Cannot reach the API server at " + API_BASE_URL
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Tab 1: Medical Term Extraction
# ---------------------------------------------------------------------------

def extract_medical_info(image_path: str | None) -> tuple[str, str]:
    """Upload the image to /extract and return (extracted_content, metadata).

    Returns a tuple of two strings so Gradio can display them in separate
    output components.
    """
    if image_path is None:
        return "Please upload a medical document image.", ""

    try:
        with open(image_path, "rb") as fh:
            files = {"file": ("upload.png", fh, "image/png")}
            resp = requests.post(_api_url("/extract"), files=files, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        content: str = data.get("extracted_content", "")
        latency: float = data.get("inference_latency_seconds", 0.0)
        model_info: dict[str, Any] = data.get("model_info", {})

        meta_lines = [
            f"Inference latency : {latency:.2f} s",
            f"Model             : {model_info.get('model_name', 'N/A')}",
            f"Precision         : {model_info.get('precision', 'N/A')}",
            f"LoRA adapter      : {model_info.get('lora_adapter', 'N/A')}",
            f"Device            : {model_info.get('device', 'N/A')}",
            f"Timestamp         : {data.get('timestamp', 'N/A')}",
        ]
        return content, "\n".join(meta_lines)

    except requests.ConnectionError:
        return "ERROR: Cannot connect to the API server.", ""
    except requests.HTTPError as exc:
        return f"ERROR: API returned {exc.response.status_code}: {exc.response.text}", ""
    except Exception as exc:
        return f"ERROR: {exc}", ""


# ---------------------------------------------------------------------------
# Tab 2: Custom Analysis
# ---------------------------------------------------------------------------

def analyze_image(image_path: str | None, prompt: str) -> tuple[str, str]:
    """Upload the image and prompt to /analyze, return (analysis, metadata)."""
    if image_path is None:
        return "Please upload a medical document image.", ""

    try:
        with open(image_path, "rb") as fh:
            files = {"file": ("upload.png", fh, "image/png")}
            form_data = {}
            if prompt and prompt.strip():
                form_data["prompt"] = prompt.strip()
            resp = requests.post(
                _api_url("/analyze"), files=files, data=form_data, timeout=300
            )
        resp.raise_for_status()
        data = resp.json()

        analysis: str = data.get("analysis", "")
        latency: float = data.get("inference_latency_seconds", 0.0)
        prompt_used: str = data.get("prompt_used", "")
        model_info: dict[str, Any] = data.get("model_info", {})

        meta_lines = [
            f"Inference latency : {latency:.2f} s",
            f"Prompt used       : {prompt_used[:120]}{'...' if len(prompt_used) > 120 else ''}",
            f"Model             : {model_info.get('model_name', 'N/A')}",
            f"Precision         : {model_info.get('precision', 'N/A')}",
            f"Device            : {model_info.get('device', 'N/A')}",
            f"Timestamp         : {data.get('timestamp', 'N/A')}",
        ]
        return analysis, "\n".join(meta_lines)

    except requests.ConnectionError:
        return "ERROR: Cannot connect to the API server.", ""
    except requests.HTTPError as exc:
        return f"ERROR: API returned {exc.response.status_code}: {exc.response.text}", ""
    except Exception as exc:
        return f"ERROR: {exc}", ""


# ---------------------------------------------------------------------------
# Tab 3: Benchmarks
# ---------------------------------------------------------------------------

def load_benchmarks() -> str:
    """Fetch all benchmarks from /benchmark and return a formatted string."""
    try:
        resp = requests.get(_api_url("/benchmark"), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        files_loaded: list[str] = data.get("files_loaded", [])
        benchmarks: dict[str, Any] = data.get("benchmarks", {})

        sections: list[str] = []
        sections.append(f"Benchmark files loaded: {len(files_loaded)}")
        sections.append("=" * 72)

        for name, content in benchmarks.items():
            sections.append(f"\n--- {name} ---")
            if isinstance(content, dict):
                sections.append(json.dumps(content, indent=2, default=str))
            else:
                sections.append(str(content))

        return "\n".join(sections)

    except requests.ConnectionError:
        return "ERROR: Cannot connect to the API server at " + API_BASE_URL
    except requests.HTTPError as exc:
        return f"ERROR: API returned {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="Medical Vision Pipeline Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Medical Vision Pipeline Demo

            **Model:** Qwen2.5-VL-7B-Instruct + LoRA fine-tuning + INT4 (NF4) quantization

            This demo connects to the FastAPI backend that serves a Qwen2.5-VL-7B
            vision-language model fine-tuned with LoRA adapters for medical document
            understanding. The model is quantized to 4-bit (NF4 via bitsandbytes)
            for efficient inference on a single NVIDIA GB10 GPU.

            Upload a medical document image (prescription, lab report, radiology
            report, discharge summary, etc.) and the model will extract structured
            clinical information.
            """
        )

        api_status = gr.Textbox(
            label="API Status",
            value=_check_api,
            interactive=False,
            every=30,
        )

        # -- Tab 1: Medical Term Extraction --------------------------------
        with gr.Tab("Medical Term Extraction"):
            gr.Markdown(
                "Upload a medical document image to extract diagnoses, "
                "medications, lab values, and vital signs."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    extract_image = gr.Image(
                        label="Upload Medical Document",
                        type="filepath",
                    )
                    extract_btn = gr.Button("Extract Medical Information", variant="primary")
                with gr.Column(scale=2):
                    extract_output = gr.Textbox(
                        label="Extracted Content",
                        lines=20,
                        show_copy_button=True,
                    )
                    extract_meta = gr.Textbox(
                        label="Request Metadata",
                        lines=6,
                        interactive=False,
                    )

            extract_btn.click(
                fn=extract_medical_info,
                inputs=[extract_image],
                outputs=[extract_output, extract_meta],
            )

        # -- Tab 2: Custom Analysis ----------------------------------------
        with gr.Tab("Custom Analysis"):
            gr.Markdown(
                "Upload a medical image and provide a custom prompt. "
                "Leave the prompt blank to use the default extraction prompt."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    analyze_image = gr.Image(
                        label="Upload Medical Document",
                        type="filepath",
                    )
                    analyze_prompt = gr.Textbox(
                        label="Custom Prompt (optional)",
                        placeholder=(
                            "e.g., List all medications with dosages and "
                            "administration routes."
                        ),
                        lines=4,
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    analyze_output = gr.Textbox(
                        label="Analysis Result",
                        lines=20,
                        show_copy_button=True,
                    )
                    analyze_meta = gr.Textbox(
                        label="Request Metadata",
                        lines=6,
                        interactive=False,
                    )

            analyze_btn.click(
                fn=analyze_image,
                inputs=[analyze_image, analyze_prompt],
                outputs=[analyze_output, analyze_meta],
            )

        # -- Tab 3: Benchmarks --------------------------------------------
        with gr.Tab("Benchmarks"):
            gr.Markdown(
                "Load all pipeline benchmark data collected during model "
                "optimization and evaluation phases."
            )
            bench_btn = gr.Button("Load Benchmark Data", variant="secondary")
            bench_output = gr.Textbox(
                label="Benchmark Results",
                lines=30,
                show_copy_button=True,
            )

            bench_btn.click(
                fn=load_benchmarks,
                inputs=[],
                outputs=[bench_output],
            )

    return demo


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Gradio demo server."""
    demo = build_demo()
    logger.info("Starting Gradio demo on %s:%d", GRADIO_HOST, GRADIO_PORT)
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
