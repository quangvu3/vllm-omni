#!/usr/bin/env bash
# Build (and optionally push) the Z-Image RunPod Docker image.
#
# Usage:
#   ./runpod/build_and_push.sh --registry docker.io/yourname --tag v1.0.0 [--image z-image-runpod] [--push]
#
# Options:
#   --registry  <registry/namespace>   e.g. docker.io/yourname  (required)
#   --image     <image-name>           default: z-image-runpod
#   --tag       <tag>                  default: latest
#   --push                             push both :<tag> and :latest after build
#   --help                             show this message

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGISTRY=""
IMAGE="z-image-runpod"
TAG="latest"
PUSH=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry)  REGISTRY="$2";  shift 2 ;;
        --image)     IMAGE="$2";     shift 2 ;;
        --tag)       TAG="$2";       shift 2 ;;
        --push)      PUSH=true;      shift   ;;
        --help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$REGISTRY" ]]; then
    echo "Error: --registry is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths — build context is always the repo root so COPY . . works correctly.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL_IMAGE="${REGISTRY}/${IMAGE}"
TAGGED_IMAGE="${FULL_IMAGE}:${TAG}"
LATEST_IMAGE="${FULL_IMAGE}:latest"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "==> Build context : ${REPO_ROOT}"
echo "==> Dockerfile    : runpod/Dockerfile"
echo "==> Image         : ${TAGGED_IMAGE}"

docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${TAGGED_IMAGE}" \
    "${REPO_ROOT}"

# Always tag :latest as well (skip if tag IS latest to avoid duplicate work).
if [[ "${TAG}" != "latest" ]]; then
    echo "==> Tagging :latest"
    docker tag "${TAGGED_IMAGE}" "${LATEST_IMAGE}"
fi

echo "==> Build complete: ${TAGGED_IMAGE}"

# ---------------------------------------------------------------------------
# Optional push
# ---------------------------------------------------------------------------
if [[ "${PUSH}" == true ]]; then
    echo "==> Pushing ${TAGGED_IMAGE}"
    docker push "${TAGGED_IMAGE}"

    if [[ "${TAG}" != "latest" ]]; then
        echo "==> Pushing ${LATEST_IMAGE}"
        docker push "${LATEST_IMAGE}"
    fi

    echo "==> Push complete."
fi
