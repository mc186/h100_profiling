#!/usr/bin/env python3
"""Batch-size perf sweep, INFERENCE-ONLY wall-clock (no profiler, no JSON).

Each workload is loaded ONCE; then for each batch size we run a warmup pass
(excluded) and time the inference call via CUDA events. This isolates the
compute/batch-scaling curve from one-time model-load overhead. Appends rows to
perf_sweep.csv; stops a workload at its first OOM.
"""
import torch, gc, os, csv, sys, shutil

OUT = "/workspace/results"
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/perf_sweep.csv"
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
DTYPE = torch.bfloat16


def clear_hf_cache():
    for d in ("/workspace/.cache/huggingface/hub", os.path.expanduser("~/.cache/huggingface/hub")):
        if os.path.exists(d):
            try:
                shutil.rmtree(d); os.makedirs(d)
            except Exception:
                pass


# ---- builders: load once, return an infer(batch) closure (inference only) ----

def build_resnet():
    import torchvision.models as models
    m = models.resnet50(weights="IMAGENET1K_V1").to("cuda").eval().to(DTYPE)
    def infer(b):
        x = torch.randn(b, 3, 224, 224, dtype=DTYPE, device="cuda")
        with torch.no_grad():
            m(x)
    return infer


def _sd_pipe(cls, model_id, **kw):
    pipe = cls.from_pretrained(model_id, torch_dtype=DTYPE, **kw).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_unet_sd():
    from diffusers import StableDiffusionPipeline
    pipe = _sd_pipe(StableDiffusionPipeline, "runwayml/stable-diffusion-v1-5")
    def infer(b):
        with torch.no_grad():
            pipe(["a photo of an astronaut riding a horse"] * b,
                 num_inference_steps=20, guidance_scale=7.5)
    return infer


def build_sdxl():
    from diffusers import StableDiffusionXLPipeline
    pipe = _sd_pipe(StableDiffusionXLPipeline, "stabilityai/stable-diffusion-xl-base-1.0")
    def infer(b):
        with torch.no_grad():
            pipe(["a professional photo of an astronaut riding a horse"] * b,
                 num_inference_steps=20)
    return infer


def build_dit():
    from diffusers import DiTPipeline, AutoencoderKL
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",
                                        subfolder="vae", torch_dtype=DTYPE).to("cuda")
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", vae=vae,
                                       torch_dtype=DTYPE).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    def infer(b):
        with torch.no_grad():
            pipe(class_labels=[0] * b, num_inference_steps=20)
    return infer


def build_llama(seq_length):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "NousResearch/Meta-Llama-3-8B"
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=DTYPE, device_map="cuda")
    base = ("The quick brown fox jumps over the lazy dog. "
            "This is a test sentence for language model profiling. ")
    text = base * (seq_length // len(tok.encode(base)) + 1)
    def infer(b):
        inp = tok([text] * b, return_tensors="pt", padding=True,
                  truncation=True, max_length=seq_length).to("cuda")
        with torch.no_grad():
            model(**inp)
    return infer


def build_cogvideox(num_frames, height, width, model_id="THUDM/CogVideoX-2b"):
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    def infer(b):
        with torch.no_grad():
            pipe(["A cat playing piano in a cozy room with warm lighting"] * b,
                 num_frames=num_frames, height=height, width=width, num_inference_steps=20)
    return infer


BUILDERS = {
    "resnet-50": build_resnet,
    "unet-sd": build_unet_sd,
    "sdxl": build_sdxl,
    "dit-xl": build_dit,
    "llama-8b-1k": lambda: build_llama(1024),
    "llama-8b-4k": lambda: build_llama(4096),
    "llama-8b-16k": lambda: build_llama(16384),
    "llama-8b-64k": lambda: build_llama(65536),
    "unet-3d": lambda: build_cogvideox(16, 480, 720),
    "cogvideox-16f-480p": lambda: build_cogvideox(16, 480, 720),
    "cogvideox-49f-480p": lambda: build_cogvideox(49, 480, 720),
    "cogvideox-81f-480p": lambda: build_cogvideox(81, 480, 720),
    "cogvideox-49f-720p": lambda: build_cogvideox(49, 720, 1280),
}

ORDER = list(BUILDERS.keys())


def time_infer(infer, b):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); infer(b); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def main():
    models = sys.argv[1:] or ORDER
    write_header = not os.path.exists(CSV)
    f = open(CSV, "a", newline=""); w = csv.writer(f)
    if write_header:
        w.writerow(["model", "batch", "infer_ms", "status"]); f.flush()

    for m in models:
        if m not in BUILDERS:
            print(f"SKIP {m}", flush=True); continue
        print(f"=== {m} ===", flush=True)
        try:
            infer = BUILDERS[m]()   # load once
        except Exception as ex:
            w.writerow([m, "", "", f"load_error:{str(ex)[:80]}"]); f.flush()
            print(f"  LOAD ERROR {str(ex)[:100]}", flush=True); clear_hf_cache(); cleanup(); continue

        for b in BATCHES:
            try:
                infer(b); torch.cuda.synchronize()      # warmup (excluded)
                ms = time_infer(infer, b)               # timed
                w.writerow([m, b, f"{ms:.2f}", "success"]); f.flush()
                print(f"  b{b:<4} {ms:9.1f} ms", flush=True)
            except torch.cuda.OutOfMemoryError:
                w.writerow([m, b, "", "oom"]); f.flush()
                print(f"  b{b:<4} OOM -> stop", flush=True); cleanup(); break
            except Exception as ex:
                w.writerow([m, b, "", f"error:{str(ex)[:80]}"]); f.flush()
                print(f"  b{b:<4} ERROR {str(ex)[:80]}", flush=True); cleanup(); break
            cleanup()

        del infer; clear_hf_cache(); cleanup()

    f.close(); print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
