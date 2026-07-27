#!/usr/bin/env python3
"""Batch-size perf sweep, wall-clock only (no profiler, no trace/kernels JSON).

For every workload, times end-to-end GPU latency per batch via CUDA events and
appends a row to perf_sweep.csv. Stops a workload at its first OOM (curve tail),
then moves on. Clears the HF cache between model families to stay under the
volume quota.
"""
import torch, gc, os, csv, sys, shutil
from run_model import MODEL_RUNNERS

OUT = "/workspace/results"
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/perf_sweep.csv"

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

# Grouped so we clear the HF cache once per family, not per row.
ORDER = [
    "resnet-50",
    "unet-sd", "sdxl", "dit-xl",
    "llama-8b-1k", "llama-8b-4k", "llama-8b-16k", "llama-8b-64k",
    "unet-3d",
    "cogvideox-16f-480p", "cogvideox-49f-480p", "cogvideox-81f-480p", "cogvideox-49f-720p",
]


def clear_cache():
    for d in ("/workspace/.cache/huggingface/hub", os.path.expanduser("~/.cache/huggingface/hub")):
        if os.path.exists(d):
            try:
                shutil.rmtree(d); os.makedirs(d)
            except Exception:
                pass


def time_runner(runner, batch):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    runner(batch)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def main():
    models = sys.argv[1:] or ORDER
    write_header = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["model", "batch", "total_ms", "status"]); f.flush()

    for m in models:
        if m not in MODEL_RUNNERS:
            print(f"SKIP unknown {m}", flush=True); continue
        print(f"=== {m} ===", flush=True)
        runner = MODEL_RUNNERS[m]
        for b in BATCHES:
            try:
                ms = time_runner(runner, b)
                w.writerow([m, b, f"{ms:.2f}", "success"]); f.flush()
                print(f"  b{b:<4} {ms:9.1f} ms", flush=True)
            except torch.cuda.OutOfMemoryError:
                w.writerow([m, b, "", "oom"]); f.flush()
                print(f"  b{b:<4} OOM -> stop", flush=True)
                cleanup(); break
            except Exception as e:
                w.writerow([m, b, "", f"error:{str(e)[:80]}"]); f.flush()
                print(f"  b{b:<4} ERROR {str(e)[:80]}", flush=True)
                cleanup(); break
            cleanup()
        clear_cache()   # free quota before the next family
        cleanup()

    f.close()
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
