# RunPod Serverless Deployment

This guide covers deploying a vllm-omni model (e.g. `Tongyi-MAI/Z-Image-Turbo`) as a
[RunPod](https://runpod.io) serverless endpoint.

The `runpod/` directory in the repository contains everything needed:

| File | Purpose |
|---|---|
| `runpod/Dockerfile` | Container image built on top of `vllm/vllm-openai` |
| `runpod/handler.py` | RunPod serverless worker — starts vllm-omni and proxies jobs |
| `runpod/build_and_push.sh` | Helper script to build and push to Docker Hub |

---

## Prerequisites

- Docker with NVIDIA container toolkit installed locally
- A [Docker Hub](https://hub.docker.com) account (or any OCI registry)
- A RunPod account with API access

---

## Step 1 — Build and Push the Image

The build context must be the **repository root** so that `COPY . .` captures the full
vllm-omni source.

```bash
cd /path/to/vllm-omni
docker login docker.io

chmod +x runpod/build_and_push.sh
./runpod/build_and_push.sh \
  --registry docker.io/YOURNAME \
  --image    z-image-runpod \
  --tag      v1.0.0 \
  --push
```

The script tags both `v1.0.0` and `latest`, then pushes both when `--push` is given.

**Build arguments** (override with `--build-arg`):

| ARG | Default |
|---|---|
| `VLLM_BASE_IMAGE` | `vllm/vllm-openai` |
| `VLLM_BASE_TAG` | `v0.16.0` |

First build takes 15–20 minutes (large base image). Subsequent builds are fast thanks to
layer caching.

---

## Step 2 — Create a Network Volume (model cache)

A network volume lets workers share downloaded weights across cold starts, cutting startup
time from ~10 minutes down to ~2 minutes after the first run.

RunPod Console → **Storage › Network Volumes › New**

| Setting | Value |
|---|---|
| Name | `z-image-model-cache` |
| Size | 50 GB (weights ≈ 20 GB + overhead) |
| Region | Match the GPU region you intend to use |

---

## Step 3 — Create a Serverless Template

RunPod Console → **Serverless › Templates › New**

| Setting | Value |
|---|---|
| Container Image | `docker.io/YOURNAME/z-image-runpod:v1.0.0` |
| Container Disk | 20 GB |

Environment variables:

| Variable | Value |
|---|---|
| `MODEL_NAME` | `Tongyi-MAI/Z-Image-Turbo` |
| `GPU_MEMORY_UTILIZATION` | `0.95` |
| `VLLM_PORT` | `8000` |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` |
| `HF_TOKEN` | `hf_xxx` _(optional — model is currently public)_ |

---

## Step 4 — Create a Serverless Endpoint

RunPod Console → **Serverless › Endpoints › New**

| Setting | Value |
|---|---|
| Template | the template created above |
| GPU type | NVIDIA RTX 4090 (24 GB) |
| Max Workers | 2–3 |
| Idle Timeout | 60 s |
| Execution Timeout | 600 s |
| Network Volume | `z-image-model-cache` mounted at `/runpod-volume` |

!!! note "Memory flags"
    The handler starts vllm-omni with `--vae-use-slicing --vae-use-tiling` automatically,
    which keeps peak VRAM within 24 GB for images up to 1024×1024.

---

## Step 5 — Send a Job

```bash
ENDPOINT_ID="your-endpoint-id"
API_KEY="your-runpod-api-key"

curl -X POST \
  "https://api.runpod.io/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A photorealistic autumn fox",
      "size": "1024x1024",
      "seed": 42
    }
  }' \
  | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
open('out.png','wb').write(base64.b64decode(d['output']['image']))
print('Saved out.png — status:', d['status'])
"
```

Or with Python using the RunPod SDK:

```python
import runpod
import base64

runpod.api_key = "your-runpod-api-key"
endpoint = runpod.Endpoint("your-endpoint-id")

result = endpoint.run_sync({
    "prompt": "A photorealistic autumn fox",
    "size": "1024x1024",
    "seed": 42,
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
| `num_inference_steps` | int | model default | Diffusion steps |
| `seed` | int | — | Random seed for reproducibility |
| `negative_prompt` | string | — | What to avoid |
| `guidance_scale` | float | model default | Classifier-free guidance scale |
| `true_cfg_scale` | float | model default | True CFG scale (Z-Image specific) |

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

## Using an Alternate Transformer (Beyond Reality fine-tune)

`linoyts/beyond-reality-z-image-diffusers` is a fine-tuned transformer that produces
highly photorealistic images. It uses the same `ZImageTransformer2DModel` architecture as
`Tongyi-MAI/Z-Image-Turbo` but has different weights. The VAE, text encoder, tokenizer,
and scheduler are still loaded from the base model.

### Setup

Add `TRANSFORMER_MODEL` to your RunPod template environment variables:

| Variable | Value |
|---|---|
| `TRANSFORMER_MODEL` | `linoyts/beyond-reality-z-image-diffusers` |

Increase the network volume size to accommodate both model sets:

| Setting | Value |
|---|---|
| Size | 35 GB (20 GB base + 12.3 GB fine-tune + overhead) |

### Recommended Inference Parameters

This fine-tune is designed for **classifier-free guidance disabled** (`guidance_scale=0.0`)
and works well with ~10 inference steps:

```bash
curl -X POST \
  "https://api.runpod.io/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a photorealistic autumn fox",
      "size": "1024x1024",
      "num_inference_steps": 10,
      "guidance_scale": 0.0,
      "seed": 42
    }
  }'
```

### Local / Online Serving

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni \
  --transformer-model linoyts/beyond-reality-z-image-diffusers
```

---

## Cold Start Behaviour

| Scenario | Duration | What happens |
|---|---|---|
| First cold start (no cache) | 8–12 min | Downloads ~20 GB weights to network volume |
| Subsequent cold starts (cached) | 2–3 min | Loads weights from `/runpod-volume` |
| Warm worker (idle < 60 s) | ~0 s | vllm-omni already running |

---

## Troubleshooting

**Job times out on first run**

The default execution timeout (600 s) may not be enough when downloading weights on a
slow connection. Either increase the timeout in the endpoint settings or pre-seed the
network volume by running a warm-up job with a simple prompt before going to production.

**Out-of-memory errors**

- Reduce image size: `"size": "512x512"`
- Reduce `GPU_MEMORY_UTILIZATION` to `0.90`
- Ensure no other process is using the GPU in the same worker

**`vllm-omni server did not become healthy` error in logs**

The vllm-omni subprocess failed to start. Check the full worker logs in the RunPod
console — the most common cause is a missing or mismatched `MODEL_NAME` environment
variable.

**Model re-downloaded on every cold start**

Confirm the network volume is mounted at `/runpod-volume` in the endpoint settings.
The handler only uses the cache path when that directory exists.
