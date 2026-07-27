#!/usr/bin/env python3
"""
H100 PyTorch Profiler for Performance Characterization

Captures per-layer runtime, kernel names, and utilization metrics
for all models across all batch sizes.

Usage:
    python profile_h100.py --model <model_name> --batch <batch_size>

Outputs:
    - trace_{model}_b{batch}.json: Chrome trace (visualize in chrome://tracing)
    - kernels_{model}_b{batch}.json: Per-kernel statistics (machine-readable)
    - profile_{model}_b{batch}.txt: Human-readable table
"""

import torch
import torch.profiler as profiler
import argparse
import json
import sys
import os

# Add parent directory to path for run_model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_model import MODEL_RUNNERS

def profile_model(model_name, batch_size):
    """Profile a model with PyTorch profiler"""

    print(f"Profiling {model_name} at batch={batch_size}")

    if model_name not in MODEL_RUNNERS:
        raise ValueError(f"Unknown model: {model_name}")

    runner = MODEL_RUNNERS[model_name]

    # Warmup run
    print("  Warmup run...")
    try:
        runner(batch_size)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"  ✗ OOM during warmup - batch size {batch_size} too large")
        raise

    # Clear cache
    torch.cuda.empty_cache()

    # Profile with PyTorch profiler
    print("  Profiling run...")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        runner(batch_size)
        torch.cuda.synchronize()

    # Export Chrome trace
    trace_file = f"trace_{model_name}_b{batch_size}.json"
    prof.export_chrome_trace(trace_file)
    print(f"  ✓ Saved: {trace_file}")

    # Get human-readable table
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=100
    )
    table_file = f"profile_{model_name}_b{batch_size}.txt"
    with open(table_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Batch: {batch_size}\n")
        f.write("="*80 + "\n\n")
        f.write(table)
    print(f"  ✓ Saved: {table_file}")

    # Extract per-kernel stats (machine-readable)
    events = prof.key_averages()
    results = []

    for evt in events:
        if evt.device_type == profiler.DeviceType.CUDA:
            # Classify kernel as attention vs dense
            kernel_name = evt.key.lower()
            if any(x in kernel_name for x in ['attention', 'attn', 'qkv', 'softmax']):
                op_class = 'attention'
            elif any(x in kernel_name for x in ['gemm', 'conv', 'matmul', 'linear', 'mlp']):
                op_class = 'dense'
            else:
                op_class = 'other'

            results.append({
                'kernel_name': evt.key,
                'op_class': op_class,
                'cuda_time_us': evt.cuda_time_total,
                'count': evt.count,
                'avg_time_us': evt.cuda_time_total / evt.count if evt.count > 0 else 0,
                'cpu_time_us': evt.cpu_time_total,
                'self_cuda_time_us': evt.self_cuda_time_total,
            })

    kernels_file = f"kernels_{model_name}_b{batch_size}.json"
    with open(kernels_file, "w") as f:
        json.dump({
            'model': model_name,
            'batch': batch_size,
            'kernels': results
        }, f, indent=2)
    print(f"  ✓ Saved: {kernels_file}")

    # Summary stats
    total_cuda_time = sum(r['cuda_time_us'] for r in results)
    attention_time = sum(r['cuda_time_us'] for r in results if r['op_class'] == 'attention')
    dense_time = sum(r['cuda_time_us'] for r in results if r['op_class'] == 'dense')

    print(f"\n  Summary:")
    print(f"    Total CUDA time: {total_cuda_time/1000:.2f} ms")
    if total_cuda_time > 0:
        print(f"    Attention: {attention_time/1000:.2f} ms ({100*attention_time/total_cuda_time:.1f}%)")
        print(f"    Dense: {dense_time/1000:.2f} ms ({100*dense_time/total_cuda_time:.1f}%)")
        print(f"    Other: {(total_cuda_time-attention_time-dense_time)/1000:.2f} ms ({100*(total_cuda_time-attention_time-dense_time)/total_cuda_time:.1f}%)")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile models on H100 with PyTorch profiler"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_RUNNERS.keys(),
        help="Model to profile"
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Batch size"
    )
    args = parser.parse_args()

    print("="*60)
    print(f"H100 Performance Profiling")
    print(f"Model: {args.model}")
    print(f"Batch: {args.batch}")
    print("="*60)

    try:
        profile_model(args.model, args.batch)
        print("\n" + "="*60)
        print("✓ Profiling complete!")
        print("="*60)
    except torch.cuda.OutOfMemoryError:
        print("\n" + "="*60)
        print(f"✗ Out of Memory - batch size {args.batch} too large for {args.model}")
        print("="*60)
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Error: {e}")
        print("="*60)
        raise
