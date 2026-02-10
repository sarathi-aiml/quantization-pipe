"""End-to-end tests for the Medical Vision API server.

This script:
    1. Starts the FastAPI server as a subprocess.
    2. Waits for the /health endpoint to report the model as loaded.
    3. Creates a synthetic medical document test image.
    4. Exercises every API endpoint (/extract, /analyze, /health, /benchmark).
    5. Validates responses contain expected medical information.
    6. Prints a test results summary.
    7. Tears down the server subprocess.

Usage
-----
    python tests/test_api.py          # from the project root
    python -m tests.test_api          # from the project root

The script exits with code 0 if all tests pass, or 1 if any test fails.
"""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_HOST = "127.0.0.1"
API_PORT = 8000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
STARTUP_TIMEOUT_SECONDS = 600  # model loading can take several minutes
POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_api")

# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

class TestResult:
    """Simple container for one test's outcome."""

    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Test image generation
# ---------------------------------------------------------------------------

def create_test_medical_image() -> str:
    """Create a synthetic medical document image and return its file path.

    The image simulates a patient assessment document with diagnoses,
    medications, vital signs, and lab values -- realistic enough for the
    model to extract structured medical information.
    """
    width, height = 900, 1100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font; fall back to default if unavailable.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
        font_title = font

    y = 30
    line_height = 24

    def write(text: str, f: ImageFont.FreeTypeFont | None = None, indent: int = 40) -> None:
        nonlocal y
        draw.text((indent, y), text, fill="black", font=f or font)
        y += line_height

    def blank(n: int = 1) -> None:
        nonlocal y
        y += line_height * n

    # --- Document content ---------------------------------------------------
    draw.text((width // 2 - 180, y), "GENERAL HOSPITAL", fill="black", font=font_title)
    y += 32
    draw.text((width // 2 - 200, y), "PATIENT ASSESSMENT DOCUMENT", fill="black", font=font_bold)
    y += 36
    draw.line([(30, y), (width - 30, y)], fill="black", width=2)
    y += 16

    write("Patient: John A. Smith          DOB: 03/15/1958", font_bold)
    write("MRN: 2024-87432                 Date: 01/20/2025")
    write("Attending: Dr. Maria Gonzalez, MD  Dept: Internal Medicine")
    blank()

    draw.line([(30, y), (width - 30, y)], fill="gray", width=1)
    y += 8
    write("CHIEF COMPLAINT:", font_bold)
    write("Chest pain radiating to left arm x 3 hours, diaphoresis")
    blank()

    write("DIAGNOSES:", font_bold)
    write("1. Acute Myocardial Infarction (STEMI) - anterior wall")
    write("2. Hypertension, Stage 2, uncontrolled")
    write("3. Type 2 Diabetes Mellitus (HbA1c 9.1%)")
    write("4. Hyperlipidemia (LDL 188 mg/dL)")
    write("5. Chronic Kidney Disease Stage 3a (eGFR 48 mL/min)")
    blank()

    write("VITAL SIGNS:", font_bold)
    write("BP: 172/98 mmHg    HR: 108 bpm    RR: 24/min")
    write("Temp: 99.1 F       SpO2: 91% on RA    Weight: 210 lbs")
    blank()

    write("MEDICATIONS ORDERED:", font_bold)
    write("- Aspirin 325 mg PO STAT, then 81 mg PO QD")
    write("- Clopidogrel 600 mg PO loading dose")
    write("- Heparin 60 units/kg IV bolus, then 12 units/kg/hr")
    write("- Metoprolol 5 mg IV q5min x 3, then 25 mg PO BID")
    write("- Atorvastatin 80 mg PO QHS")
    write("- Nitroglycerin 0.4 mg SL PRN chest pain q5min x 3")
    write("- Morphine 2-4 mg IV q5-15min PRN pain")
    write("- Metformin 1000 mg PO BID (hold if Cr > 1.5)")
    write("- Lisinopril 10 mg PO QD (start post-cath)")
    blank()

    write("LABORATORY RESULTS:", font_bold)
    write("Troponin I: 4.82 ng/mL (H)    Ref: < 0.04 ng/mL")
    write("BNP:        892 pg/mL  (H)    Ref: < 100 pg/mL")
    write("Creatinine: 1.6 mg/dL  (H)    Ref: 0.6-1.2 mg/dL")
    write("Glucose:    238 mg/dL  (H)    Ref: 70-100 mg/dL")
    write("Potassium:  5.2 mEq/L  (H)    Ref: 3.5-5.0 mEq/L")
    write("WBC:        12.4 K/uL  (H)    Ref: 4.5-11.0 K/uL")
    write("Hemoglobin: 11.8 g/dL  (L)    Ref: 13.5-17.5 g/dL")
    blank()

    write("PLAN:", font_bold)
    write("- Emergent cardiac catheterization with PCI")
    write("- Admit to CCU, continuous telemetry monitoring")
    write("- Repeat troponin q6h x 3")
    write("- Echocardiogram within 24 hours")
    write("- Endocrinology consult for DM management")
    blank()
    draw.line([(30, y), (width - 30, y)], fill="black", width=2)
    y += 12
    write("Electronically signed: Dr. Maria Gonzalez, MD  01/20/2025 14:32")

    # Save to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, "PNG")
    tmp.close()
    logger.info("Test image created at %s (%dx%d)", tmp.name, width, height)
    return tmp.name


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server() -> subprocess.Popen:
    """Start the FastAPI server as a subprocess and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.server:app",
        "--host", API_HOST,
        "--port", str(API_PORT),
        "--workers", "1",
    ]
    logger.info("Starting API server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # own process group for clean teardown
    )
    return proc


def wait_for_health(timeout: int = STARTUP_TIMEOUT_SECONDS) -> bool:
    """Poll /health until model_loaded is True or timeout expires.

    Returns True if the server became healthy, False otherwise.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded"):
                    logger.info("Server healthy -- model loaded.")
                    return True
                logger.info(
                    "Server reachable but model not loaded yet (VRAM: %.0f MB). Waiting ...",
                    data.get("current_vram_usage_mb", 0),
                )
        except requests.ConnectionError:
            logger.info("Server not reachable yet. Waiting ...")
        except Exception as exc:
            logger.warning("Health check error: %s", exc)
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the server subprocess and its process group."""
    if proc.poll() is not None:
        return
    logger.info("Stopping API server (PID %d) ...", proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Server did not stop gracefully; sending SIGKILL.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def test_health() -> TestResult:
    """Test GET /health returns expected fields."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        required_keys = {"model_loaded", "gpu_available", "gpu_name", "current_vram_usage_mb", "timestamp"}
        missing = required_keys - set(data.keys())
        if missing:
            return TestResult("GET /health", False, f"Missing keys: {missing}")
        if not data["model_loaded"]:
            return TestResult("GET /health", False, "model_loaded is False")
        return TestResult(
            "GET /health",
            True,
            f"OK -- GPU: {data['gpu_name']}, VRAM: {data['current_vram_usage_mb']:.0f} MB",
            details=json.dumps(data, indent=2),
        )
    except Exception as exc:
        return TestResult("GET /health", False, str(exc))


def test_extract(image_path: str) -> TestResult:
    """Test POST /extract with a medical document image."""
    try:
        with open(image_path, "rb") as fh:
            files = {"file": ("test_document.png", fh, "image/png")}
            resp = requests.post(
                f"{API_BASE_URL}/extract",
                files=files,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        resp.raise_for_status()
        data = resp.json()

        content: str = data.get("extracted_content", "")
        latency: float = data.get("inference_latency_seconds", 0)

        if not content or len(content) < 50:
            return TestResult(
                "POST /extract",
                False,
                f"Response too short ({len(content)} chars)",
                details=content,
            )

        # Check for expected medical terms (case-insensitive)
        content_lower = content.lower()
        expected_terms = [
            "myocardial", "hypertension", "diabetes", "aspirin",
            "metoprolol", "troponin", "creatinine",
        ]
        found = [t for t in expected_terms if t in content_lower]
        missed = [t for t in expected_terms if t not in content_lower]

        if len(found) < 3:
            return TestResult(
                "POST /extract",
                False,
                f"Only {len(found)}/{len(expected_terms)} medical terms found. "
                f"Missed: {missed}",
                details=content[:500],
            )

        return TestResult(
            "POST /extract",
            True,
            f"OK -- {len(found)}/{len(expected_terms)} terms found, "
            f"latency={latency:.2f}s, response={len(content)} chars",
            details=content[:500],
        )
    except Exception as exc:
        return TestResult("POST /extract", False, str(exc))


def test_analyze(image_path: str) -> TestResult:
    """Test POST /analyze with a custom prompt."""
    custom_prompt = (
        "List every medication mentioned in this document along with its "
        "dosage and route of administration. Format as a numbered list."
    )
    try:
        with open(image_path, "rb") as fh:
            files = {"file": ("test_document.png", fh, "image/png")}
            data_fields = {"prompt": custom_prompt}
            resp = requests.post(
                f"{API_BASE_URL}/analyze",
                files=files,
                data=data_fields,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        resp.raise_for_status()
        data = resp.json()

        analysis: str = data.get("analysis", "")
        prompt_used: str = data.get("prompt_used", "")
        latency: float = data.get("inference_latency_seconds", 0)

        if not analysis or len(analysis) < 30:
            return TestResult(
                "POST /analyze",
                False,
                f"Response too short ({len(analysis)} chars)",
                details=analysis,
            )

        # Verify that the custom prompt was used
        if custom_prompt not in prompt_used:
            return TestResult(
                "POST /analyze",
                False,
                "Custom prompt was not reflected in prompt_used field",
            )

        # Check for medication-related terms
        analysis_lower = analysis.lower()
        med_terms = ["aspirin", "metoprolol", "atorvastatin", "heparin", "mg"]
        found = [t for t in med_terms if t in analysis_lower]

        if len(found) < 2:
            return TestResult(
                "POST /analyze",
                False,
                f"Only {len(found)}/{len(med_terms)} medication terms found",
                details=analysis[:500],
            )

        return TestResult(
            "POST /analyze",
            True,
            f"OK -- {len(found)}/{len(med_terms)} med terms, "
            f"latency={latency:.2f}s, response={len(analysis)} chars",
            details=analysis[:500],
        )
    except Exception as exc:
        return TestResult("POST /analyze", False, str(exc))


def test_benchmark() -> TestResult:
    """Test GET /benchmark returns compiled benchmark data."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/benchmark", timeout=REQUEST_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        data = resp.json()

        files_loaded: list[str] = data.get("files_loaded", [])
        benchmarks: dict[str, Any] = data.get("benchmarks", {})

        if not files_loaded:
            return TestResult(
                "GET /benchmark",
                False,
                "No benchmark files loaded",
            )

        return TestResult(
            "GET /benchmark",
            True,
            f"OK -- {len(files_loaded)} files: {', '.join(files_loaded)}",
            details=json.dumps(list(benchmarks.keys()), indent=2),
        )
    except Exception as exc:
        return TestResult("GET /benchmark", False, str(exc))


def test_bad_request() -> TestResult:
    """Test that the server returns 400 for an invalid upload."""
    try:
        # Send a non-image file
        files = {"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")}
        resp = requests.post(
            f"{API_BASE_URL}/extract",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if resp.status_code == 400:
            return TestResult(
                "POST /extract (bad input)",
                True,
                "OK -- correctly returned 400",
            )
        return TestResult(
            "POST /extract (bad input)",
            False,
            f"Expected 400 but got {resp.status_code}",
        )
    except Exception as exc:
        return TestResult("POST /extract (bad input)", False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full test run."""
    print("=" * 72)
    print("  MEDICAL VISION API -- END-TO-END TEST SUITE")
    print("=" * 72)
    print()

    # 1. Create test image
    logger.info("Creating test medical document image ...")
    image_path = create_test_medical_image()

    # 2. Start server
    server_proc = start_server()

    results: list[TestResult] = []
    try:
        # 3. Wait for server readiness
        logger.info("Waiting for server to become healthy (timeout %ds) ...", STARTUP_TIMEOUT_SECONDS)
        if not wait_for_health():
            print("\nFATAL: Server did not become healthy within the timeout.")
            print("Server output (last 50 lines):")
            if server_proc.stdout:
                # Read whatever output is available
                try:
                    for line in server_proc.stdout:
                        print("  ", line.rstrip())
                except Exception:
                    pass
            stop_server(server_proc)
            sys.exit(1)

        # 4. Run tests
        logger.info("Running tests ...")
        results.append(test_health())
        results.append(test_extract(image_path))
        results.append(test_analyze(image_path))
        results.append(test_benchmark())
        results.append(test_bad_request())

    finally:
        # 5. Cleanup
        stop_server(server_proc)
        try:
            os.unlink(image_path)
        except OSError:
            pass

    # 6. Print results
    print()
    print("=" * 72)
    print("  TEST RESULTS")
    print("=" * 72)

    passed = 0
    failed = 0
    for r in results:
        status_marker = "PASS" if r.passed else "FAIL"
        print(f"  [{status_marker}]  {r.name}")
        print(f"         {r.message}")
        if r.details and not r.passed:
            # Show details only for failures
            for line in r.details.split("\n")[:10]:
                print(f"           {line}")
        print()
        if r.passed:
            passed += 1
        else:
            failed += 1

    print("-" * 72)
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    print("=" * 72)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
