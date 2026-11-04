#!/usr/bin/env python3
"""Hardware-agnostic vLLM launcher for the bundled `vllm-embed` / `vllm-main` services.

Runs inside the vLLM image (CUDA or ROCm) before exec'ing the OpenAI server, and:

* picks the model for the role from MODEL_FAMILY / VRAM_TIER (main) or EMBED_MODEL;
* picks quantization when QUANTIZATION=auto: fp8 where the GPU supports it
  (NVIDIA Ampere or newer, AMD MI300+), otherwise unquantized;
* sizes --gpu-memory-utilization from an absolute GB budget and the memory that
  is actually FREE right now (torch.cuda.mem_get_info works on CUDA, ROCm and
  unified-memory systems like DGX Spark / Strix Halo). vLLM refuses to start if
  utilization x total exceeds free memory, so a fixed fraction crash-loops
  whenever anything else holds GPU memory, and over-allocates on 128 GB APUs;
* for the main model, waits for the embedder to finish loading first so the two
  don't race for the same free memory.

Every choice can be overridden by env (GPU_MEMORY_GB, GPU_MEMORY_UTIL, QUANTIZATION=auto|fp8|none).
"""
import os
import sys
import time
import urllib.request

GIB = 1024 ** 3

# Must match src/local_models.py TIER_MODELS["vllm"]. Second value: approximate
# BF16 weight size in GiB (vision encoder included).
MAIN_MODELS = {
    "qwen": {
        "low": ("Qwen/Qwen3-VL-4B-Instruct", 9),
        "mid": ("Qwen/Qwen3-VL-8B-Instruct", 18),
        "high": ("Qwen/Qwen3-VL-32B-Instruct", 67),
        "ultra": ("Qwen/Qwen3-VL-235B-A22B-Instruct", 471),
    },
    "gemma": {
        "low": ("google/gemma-3-4b-it", 9),
        "mid": ("google/gemma-3-12b-it", 25),
        "high": ("google/gemma-3-12b-it", 25),
        "ultra": ("google/gemma-3-27b-it", 55),
    },
}
EMBED_WEIGHTS_GB = 1.5
HEADROOM_GB = 1.5   # left free for the CUDA/HIP context and other processes
MAIN_KV_GB = 8      # KV cache + activations for one user at MAX_MODEL_LEN
EMBED_KV_GB = 2


def log(msg: str) -> None:
    print(f"[lvs-vllm] {msg}", flush=True)


def gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        sys.exit("[lvs-vllm] No GPU visible to this container. Check GPU passthrough for your "
                 "platform (nvidia-container-toolkit, or /dev/kfd + /dev/dri for AMD).")
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
    return {
        "name": props.name,
        "free_gb": free / GIB,
        "total_gb": total / GIB,
        "rocm": torch.version.hip is not None,
        "arch": arch,
        "capability": torch.cuda.get_device_capability(0),
    }


def supports_fp8(gpu: dict) -> bool:
    if gpu["rocm"]:
        return gpu["arch"].startswith(("gfx94", "gfx95"))
    # Ampere+: native FP8 on Ada/Hopper/Blackwell, weight-only (Marlin) on Ampere.
    return gpu["capability"] >= (8, 0)


def wait_for(url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    log(f"waiting for {url} before sizing (up to {timeout_s // 60} min)...")
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            log("embedder is up")
            return
        except Exception:
            time.sleep(5)
    log("embedder still not up; continuing anyway")


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "main"
    env = os.environ
    quant = ""

    if role == "embed":
        model = env.get("EMBED_MODEL") or "BAAI/bge-m3"
        port = "8001"
        extra: list[str] = []
    else:
        family = env.get("MODEL_FAMILY", "qwen")
        family = family if family in MAIN_MODELS else "qwen"
        tier = env.get("VRAM_TIER", "mid")
        tier = tier if tier in MAIN_MODELS[family] else "mid"
        model, bf16_gb = MAIN_MODELS[family][tier]
        port = "8002"
        extra = ["--max-model-len", env.get("MAX_MODEL_LEN", "16384"), "--enforce-eager"]
        embed_url = env.get("EMBED_HEALTH_URL", "http://vllm-embed:8001/health")
        if embed_url:
            wait_for(embed_url, int(env.get("EMBED_WAIT_SECONDS", "1800")))

    gpu = gpu_info()
    arch = gpu["arch"] or "sm_" + "".join(map(str, gpu["capability"]))
    log(f"GPU: {gpu['name']} ({arch}), {gpu['free_gb']:.1f} GiB free of {gpu['total_gb']:.1f} GiB")

    if role == "embed":
        need_gb = EMBED_WEIGHTS_GB + 1
        budget_gb = EMBED_WEIGHTS_GB + EMBED_KV_GB
    else:
        quant = env.get("QUANTIZATION", "auto").strip().lower()
        if quant == "auto":
            quant = "fp8" if supports_fp8(gpu) else ""
        elif quant in ("none", "bf16", "off", "false", "0"):
            quant = ""
        # FP8 quantizes the language model only; the vision encoder stays BF16.
        weights_gb = bf16_gb * (0.55 if quant == "fp8" else 1.0)
        need_gb = weights_gb + 3
        budget_gb = weights_gb + MAIN_KV_GB

    if env.get("GPU_MEMORY_UTIL", "auto") not in ("", "auto"):
        util = float(env["GPU_MEMORY_UTIL"])
        log(f"using GPU_MEMORY_UTIL={util} from env")
    else:
        if env.get("GPU_MEMORY_GB"):
            budget_gb = float(env["GPU_MEMORY_GB"])
        alloc_gb = min(budget_gb, gpu["free_gb"] - HEADROOM_GB)
        waited = 0
        # Wait for memory instead of exiting: a restart loop looks broken and
        # gets slower with Docker's backoff, while waiting starts the model as
        # soon as whatever holds the memory lets go.
        while alloc_gb < need_gb:
            if waited % 300 == 0:
                log(f"{model} ({quant or 'bf16'}) needs ~{need_gb:.0f} GiB of GPU memory but only "
                    f"{gpu['free_gb']:.1f} GiB is free. Waiting for memory to free up; or lower "
                    f"VRAM_TIER in .env (currently {env.get('VRAM_TIER', 'mid')}) or stop other "
                    f"processes using the GPU.")
            time.sleep(30)
            waited += 30
            gpu = gpu_info()
            alloc_gb = min(budget_gb, gpu["free_gb"] - HEADROOM_GB)
        if waited:
            log(f"{gpu['free_gb']:.1f} GiB free now; starting")
        util = round(alloc_gb / gpu["total_gb"], 3)
        log(f"allocating {alloc_gb:.1f} GiB (budget {budget_gb:.1f}) -> gpu-memory-utilization {util}")

    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", "0.0.0.0", "--port", port,
        "--gpu-memory-utilization", str(util), *extra,
    ]
    if quant:
        args += ["--quantization", quant]
    log(f"starting {role}: {model} quant={quant or 'none'}")
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
