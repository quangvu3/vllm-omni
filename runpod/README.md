# RunPod Serverless Deployment

Deploy `Tongyi-MAI/Z-Image-Turbo` (with the `linoyts/beyond-reality-z-image-diffusers`
fine-tune baked in by default) as a RunPod serverless endpoint.

| File | Purpose |
|---|---|
| `Dockerfile` | Container image built on top of `vllm/vllm-openai` |
| `handler.py` | RunPod serverless worker — starts vllm-omni and proxies jobs |
| `build_and_push.sh` | Helper script to build and push to Docker Hub |

---

## Prerequisites

- Docker with NVIDIA container toolkit installed locally
- A [Docker Hub](https://hub.docker.com) account (or any OCI registry)
- A RunPod account with API access

---

## Step 1 — Build the Image

The build context must be the **repository root** so that `COPY . .` picks up the full
vllm-omni source.

```bash
cd /path/to/vllm-omni

chmod +x runpod/build_and_push.sh
./runpod/build_and_push.sh \
  --registry docker.io/YOURNAME \
  --image    z-image-realism-runpod \
  --tag      v1.0.0
```

**Build arguments** (pass with `--build-arg KEY=VALUE`, repeatable):

| ARG | Default | Description |
|---|---|---|
| `VLLM_BASE_IMAGE` | `vllm/vllm-openai` | Base image repository |
| `VLLM_BASE_TAG` | `v0.16.0` | Base image tag |

```bash
# Example: override the base image tag
./runpod/build_and_push.sh \
  --registry docker.io/YOURNAME \
  --image    z-image-realism-runpod \
  --tag      v1.0.0 \
  --build-arg VLLM_BASE_TAG=v0.8.5 \
  --push
```

First build takes 15–20 minutes. Subsequent builds are fast thanks to layer caching.

---

## Step 2 — Test Locally

Before pushing, verify the image works on your machine.

**Start the vllm-omni server** (mounts your local HF cache to skip re-downloading weights):

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v $HOME/.cache/huggingface:/runpod-volume/huggingface \
  -e HF_HOME=/runpod-volume/huggingface \
  docker.io/YOURNAME/z-image-realism-runpod:v1.0.0 \
  python -m vllm_omni.entrypoints.cli.main serve Tongyi-MAI/Z-Image-Turbo \
    --omni \
    --host 0.0.0.0 \
    --port 8000 \
    --transformer-model linoyts/beyond-reality-z-image-diffusers \
    --vae-use-slicing \
    --vae-use-tiling \
    --gpu-memory-utilization 0.95
```

**Send a test request** (in a second terminal once the server is healthy):

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a photorealistic autumn fox",
    "size": "1024x1024",
    "num_inference_steps": 10,
    "guidance_scale": 0.0
  }' \
  | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
open('test_out.png','wb').write(base64.b64decode(d['data'][0]['b64_json']))
print('Saved test_out.png')
"
```

---

## Step 3 — Push the Image

```bash
docker login docker.io

./runpod/build_and_push.sh \
  --registry docker.io/YOURNAME \
  --image    z-image-realism-runpod \
  --tag      v1.0.0 \
  --push
```

The script tags and pushes both `v1.0.0` and `latest`.

---

## Step 4 — Create a Network Volume (model cache)

A network volume lets workers share downloaded weights across cold starts, cutting startup
time from ~10 minutes to ~2 minutes after the first run.

RunPod Console → **Storage › Network Volumes › New**

| Setting | Value |
|---|---|
| Name | `z-image-model-cache` |
| Size | 40 GB (20 GB base + 12.3 GB fine-tune + overhead) |
| Region | Match the GPU region you intend to use |

---

## Step 5 — Create a Serverless Template

RunPod Console → **Serverless › Templates › New**

| Setting | Value |
|---|---|
| Container Image | `docker.io/YOURNAME/z-image-realism-runpod:v1.0.0` |
| Container Disk | 20 GB |

Environment variables (all have sensible defaults baked into the image):

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Tongyi-MAI/Z-Image-Turbo` | Base model HF repo |
| `TRANSFORMER_MODEL` | `linoyts/beyond-reality-z-image-diffusers` | Fine-tuned transformer weights |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of VRAM to use |
| `VLLM_PORT` | `8000` | Internal server port |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Required for vllm-omni |
| `HF_TOKEN` | — | HuggingFace token _(optional — models are public)_ |

To use the **base transformer only** (no fine-tune), set `TRANSFORMER_MODEL` to an empty string.

---

## Step 6 — Create a Serverless Endpoint

RunPod Console → **Serverless › Endpoints › New**

| Setting | Value |
|---|---|
| Template | the template created above |
| GPU type | NVIDIA RTX 6000 Ada / A6000 (48 GB) |
| Max Workers | 2–3 |
| Idle Timeout | 60 s |
| Execution Timeout | 600 s |
| Network Volume | `z-image-model-cache` mounted at `/runpod-volume` |

---

## Step 7 — Send a Job

```bash
ENDPOINT_ID="your-endpoint-id"
API_KEY="your-runpod-api-key"

curl -X POST \
  "https://api.runpod.io/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a photorealistic autumn fox",
      "size": "1024x1024",
      "num_inference_steps": 10,
      "guidance_scale": 0.0
    }
  }' \
  | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
open('out.png','wb').write(base64.b64decode(d['output']['image']))
print('Saved out.png — status:', d['status'])
"
```

Or with the RunPod Python SDK:

```python
import runpod, base64

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

result = endpoint.run_sync({
    "prompt": "a photorealistic autumn fox",
    "size": "1024x1024",
    "num_inference_steps": 10,
    "guidance_scale": 0.0,
})

with open("out.png", "wb") as f:
    f.write(base64.b64decode(result["image"]))
```

---

## Job Input Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Text description of the image |
| `size` | string | `"1024x1024"` | Dimensions in `WxH` format |
| `n` | int | `1` | Number of images to generate |
| `num_inference_steps` | int | model default | Diffusion steps (10 recommended for beyond-reality) |
| `guidance_scale` | float | model default | CFG scale (0.0 recommended for beyond-reality) |
| `seed` | int | — | Random seed for reproducibility |
| `negative_prompt` | string | — | What to avoid in the image |
| `true_cfg_scale` | float | — | True CFG scale (Qwen-Image specific) |

## Job Output

Single image (`n=1`):

```json
{ "image": "<base64-encoded PNG>" }
```

Multiple images (`n>1`):

```json
{ "images": ["<base64 PNG>", "<base64 PNG>", "..."] }
```

---

## Cold Start Behaviour

| Scenario | Duration | What happens |
|---|---|---|
| First cold start (no cache) | 10–15 min | Downloads ~32 GB weights to network volume |
| Subsequent cold starts (cached) | 2–3 min | Loads weights from `/runpod-volume` |
| Warm worker (idle < 60 s) | ~0 s | vllm-omni already running |

---

## Troubleshooting

**Job times out on first run**

The first cold start downloads ~32 GB. Increase the endpoint execution timeout to
900 s or pre-seed the network volume with a warm-up job before going to production.

**Out-of-memory errors**

- Reduce image size: `"size": "512x512"`
- Lower `GPU_MEMORY_UTILIZATION` to `0.90`
- Ensure no other process is using the GPU in the same worker

**`vllm-omni server did not become healthy` error in logs**

The vllm-omni subprocess failed to start. Check the full worker logs in the RunPod
console — the most common cause is a missing or mismatched `MODEL_NAME`.

**Model re-downloaded on every cold start**

Confirm the network volume is mounted at `/runpod-volume` in the endpoint settings.
The handler only activates the cache when that directory exists.
