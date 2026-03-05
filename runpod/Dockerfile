ARG VLLM_BASE_IMAGE=vllm/vllm-openai
ARG VLLM_BASE_TAG=v0.16.0
FROM ${VLLM_BASE_IMAGE}:${VLLM_BASE_TAG}

RUN apt-get update && \
    apt-get install -y ffmpeg git sox libsox-fmt-all jq && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG APP_DIR=/workspace/vllm-omni
WORKDIR ${APP_DIR}
COPY . .

RUN uv pip install --system --no-cache-dir "."
RUN uv pip install --system --no-cache-dir "runpod>=1.7.0"
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN cp runpod/handler.py /workspace/handler.py

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_PORT=8000
ENV MODEL_NAME=Tongyi-MAI/Z-Image-Turbo
ENV GPU_MEMORY_UTILIZATION=0.95
ENV TRANSFORMER_MODEL=linoyts/beyond-reality-z-image-diffusers

ENTRYPOINT []
CMD ["python", "/workspace/handler.py"]
