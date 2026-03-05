"""
RunPod serverless handler for Z-Image-Turbo via vllm-omni.

Startup: launches `vllm serve <model> --omni` as a subprocess, polls
GET /health until {"status":"healthy"} (10-minute timeout), then starts
the RunPod handler loop.

Per-job: proxies POST /v1/images/generations, returns {"image": "<b64>"} or
{"images": [...]} when n > 1.
"""

import os
import signal
import subprocess
import sys
import time

import requests
import runpod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Tongyi-MAI/Z-Image-Turbo")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")
TRANSFORMER_MODEL = os.environ.get("TRANSFORMER_MODEL", "")

# Network volume mount point — used for model caching across cold starts.
RUNPOD_VOLUME = "/runpod-volume"
HF_CACHE_DIR = os.path.join(RUNPOD_VOLUME, "huggingface")

BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"
HEALTH_URL = f"{BASE_URL}/health"
GENERATE_URL = f"{BASE_URL}/v1/images/generations"

SERVER_START_TIMEOUT = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server_proc: subprocess.Popen | None = None


def _build_serve_cmd() -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        MODEL_NAME,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(VLLM_PORT),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--vae-use-slicing",
        "--vae-use-tiling",
    ]

    # Point vllm to the network volume for model downloads when available.
    if os.path.isdir(RUNPOD_VOLUME):
        cmd += ["--download-dir", HF_CACHE_DIR]

    if TRANSFORMER_MODEL:
        cmd += ["--transformer-model", TRANSFORMER_MODEL]

    return cmd


def _start_server() -> None:
    global _server_proc

    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Point HuggingFace cache to network volume when available.
    if os.path.isdir(RUNPOD_VOLUME):
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        env["HF_HOME"] = HF_CACHE_DIR

    cmd = _build_serve_cmd()
    print(f"[handler] Starting vllm-omni server: {' '.join(cmd)}", flush=True)

    _server_proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )


def _wait_for_healthy() -> None:
    """Poll GET /health until {"status":"healthy"} or timeout."""
    deadline = time.time() + SERVER_START_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("status") == "healthy":
                    print(f"[handler] Server is healthy after {attempt} attempts.", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)

    raise RuntimeError(
        f"vllm-omni server did not become healthy within {SERVER_START_TIMEOUT}s"
    )


def _stop_server() -> None:
    global _server_proc
    if _server_proc is not None:
        try:
            os.killpg(_server_proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            _server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(_server_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _server_proc.wait()
        _server_proc = None


# ---------------------------------------------------------------------------
# RunPod job handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """
    RunPod job handler — proxies a single image-generation request.

    Accepted ``job["input"]`` fields
    --------------------------------
    prompt              str   required
    size                str   "1024x1024" (default)
    num_inference_steps int   50 (default)
    seed                int   optional
    negative_prompt     str   optional
    guidance_scale      float optional
    true_cfg_scale      float optional  (Z-Image CFG)
    n                   int   1 (default)

    Returns
    -------
    {"image": "<b64_png>"}          when n == 1
    {"images": ["<b64>", ...]}      when n > 1
    """
    job_input: dict = job.get("input", {})

    if not job_input.get("prompt"):
        return {"error": "Missing required field: prompt"}

    payload: dict = {
        "prompt": job_input["prompt"],
        "n": int(job_input.get("n", 1)),
        "size": job_input.get("size", "1024x1024"),
        "response_format": "b64_json",
    }

    # Optional diffusion parameters
    for field in (
        "num_inference_steps",
        "seed",
        "negative_prompt",
        "guidance_scale",
        "true_cfg_scale",
    ):
        if field in job_input:
            payload[field] = job_input[field]

    try:
        resp = requests.post(GENERATE_URL, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        return {"error": f"vllm-omni HTTP {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        return {"error": str(exc)}

    data = resp.json().get("data", [])
    if not data:
        return {"error": "Empty response from vllm-omni"}

    images = [item["b64_json"] for item in data if item.get("b64_json")]

    if payload["n"] == 1:
        return {"image": images[0] if images else None}
    return {"images": images}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _start_server()
    try:
        _wait_for_healthy()
    except RuntimeError as exc:
        _stop_server()
        raise SystemExit(str(exc)) from exc

    print("[handler] Registering RunPod handler.", flush=True)
    runpod.serverless.start({"handler": handler})
