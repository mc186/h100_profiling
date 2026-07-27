#!/usr/bin/env python3
"""Headless profiler for the DiT-XL and Llama gap configs.

Replicates H100_Profiling_OneByOne.ipynb's profile_model() (attention/dense
kernel split) without a notebook, so it can be driven over SSH. Writes/updates
/workspace/results/summary.json plus per-run trace_*.json and kernels_*.json.
"""
import torch
import torch.profiler as profiler
import gc, json, os, sys
from run_model import MODEL_RUNNERS

OUT = "/workspace/results"
os.makedirs(OUT, exist_ok=True)


def profile_model(model_name, batch_size, output_dir=OUT):
    runner = MODEL_RUNNERS[model_name]
    # Warmup
    try:
        runner(batch_size); torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return None
    torch.cuda.empty_cache()
    # Profile
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True,
    ) as prof:
        runner(batch_size); torch.cuda.synchronize()
    prof.export_chrome_trace(f"{output_dir}/trace_{model_name}_b{batch_size}.json")

    results = []
    for evt in prof.key_averages():
        if evt.device_type != profiler.DeviceType.CUDA:
            continue
        k = evt.key.lower()
        if any(x in k for x in ['attention', 'attn', 'qkv', 'softmax']):
            op_class = 'attention'
        elif any(x in k for x in ['gemm', 'conv', 'matmul', 'linear', 'mlp']):
            op_class = 'dense'
        else:
            op_class = 'other'
        if hasattr(evt, 'cuda_time_total'):
            t = evt.cuda_time_total
        elif hasattr(evt, 'device_time_total'):
            t = evt.device_time_total
        else:
            t = evt.self_cuda_time_total
        results.append({'kernel_name': evt.key, 'op_class': op_class,
                        'cuda_time_us': t, 'count': evt.count})

    with open(f"{output_dir}/kernels_{model_name}_b{batch_size}.json", 'w') as f:
        json.dump({'model': model_name, 'batch': batch_size, 'kernels': results}, f, indent=2)

    total = sum(r['cuda_time_us'] for r in results)
    attn = sum(r['cuda_time_us'] for r in results if r['op_class'] == 'attention')
    dense = sum(r['cuda_time_us'] for r in results if r['op_class'] == 'dense')
    return {
        'total_time_ms': total / 1000,
        'attention_time_ms': attn / 1000,
        'dense_time_ms': dense / 1000,
        'attention_pct': 100 * attn / total if total > 0 else 0,
        'dense_pct': 100 * dense / total if total > 0 else 0,
    }


def cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "dit-xl", "llama-8b-1k", "llama-8b-4k", "llama-8b-16k", "llama-8b-64k",
    ]
    batch = int(os.environ.get("PROFILE_BATCH", "1"))

    summary_path = f"{OUT}/summary.json"
    try:
        summary = json.load(open(summary_path))
    except FileNotFoundError:
        summary = []
    # Drop any prior rows for the models we're about to (re)profile
    targets = set(models)
    summary = [r for r in summary if not (r['model'] in targets and r.get('batch') == batch)]

    for m in models:
        print(f"=== {m} b{batch} ===", flush=True)
        try:
            res = profile_model(m, batch)
            if res:
                row = {'model': m, 'batch': batch, 'status': 'success', **res}
                print(f"  OK total={res['total_time_ms']:.1f}ms "
                      f"attn={res['attention_pct']:.1f}% dense={res['dense_pct']:.1f}%", flush=True)
            else:
                row = {'model': m, 'batch': batch, 'status': 'failed'}
                print("  OOM", flush=True)
        except Exception as e:
            row = {'model': m, 'batch': batch, 'status': 'error', 'error': str(e)[:200]}
            print(f"  ERROR {str(e)[:120]}", flush=True)
        summary.append(row)
        json.dump(summary, open(summary_path, 'w'), indent=2)
        cleanup()

    print("ALL_DONE", flush=True)
