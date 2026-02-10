# ---------------------------------------------------------------------------
# Dockerfile -- Medical Vision Pipeline
#
# Serves the Qwen2.5-VL-7B fine-tuned model via FastAPI (port 8000) and a
# Gradio demo UI (port 7860).  Requires an NVIDIA GPU runtime.
#
# Build:
#     docker build -t medical-vision-pipeline .
#
# Run (NVIDIA Container Toolkit required):
#     docker run --gpus all -p 8000:8000 -p 7860:7860 medical-vision-pipeline
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL maintainer="medical-vision-pipeline"
LABEL description="Qwen2.5-VL-7B + LoRA + NF4 medical document extraction API"

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        wget \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libfontconfig1 \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 -> python3.10 and pip -> pip3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy the full project
# ---------------------------------------------------------------------------
COPY . /app

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
EXPOSE 8000
EXPOSE 7860

# ---------------------------------------------------------------------------
# Health check -- probe the FastAPI /health endpoint
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/health', timeout=5); r.raise_for_status()" \
    || exit 1

# ---------------------------------------------------------------------------
# Entrypoint -- launch both FastAPI and Gradio
#
# supervisord would be cleaner for production, but a simple shell script
# keeps the image light and the two processes tightly coupled.
# ---------------------------------------------------------------------------
COPY <<'ENTRYPOINT_SCRIPT' /app/entrypoint.sh
#!/usr/bin/env bash
set -e

echo "[entrypoint] Starting FastAPI server on port 8000 ..."
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

# Wait until the API health endpoint is responsive (model loading may take
# several minutes on first start).
echo "[entrypoint] Waiting for API to become healthy ..."
for i in $(seq 1 120); do
    if python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()" 2>/dev/null; then
        echo "[entrypoint] API is healthy."
        break
    fi
    sleep 5
done

echo "[entrypoint] Starting Gradio demo on port 7860 ..."
python -m api.demo_ui &
GRADIO_PID=$!

# Forward signals to children
trap "kill $API_PID $GRADIO_PID 2>/dev/null; exit 0" SIGTERM SIGINT

# Wait for either process to exit
wait -n $API_PID $GRADIO_PID
EXIT_CODE=$?
kill $API_PID $GRADIO_PID 2>/dev/null || true
exit $EXIT_CODE
ENTRYPOINT_SCRIPT

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
