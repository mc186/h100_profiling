# Quick Start Guide

## Files Saved

All scripts are in: `~/h100_profiling/`

```
h100_profiling/
├── README.md                     # Full documentation
├── QUICK_START.md                # This file
├── run_model.py                  # Model runner (shared by both machines)
├── test_setup.sh                 # Quick setup test
├── a6000/
│   └── ncu_profile_all.sh        # A6000 NCU profiling (B=2 only)
└── h100/
    ├── profile_h100.py           # H100 PyTorch profiler
    └── run_sweep_h100.sh         # H100 full batch sweep
```

## Run on A6000 (Free - Your Machine)

```bash
# 1. Transfer to A6000
scp -r ~/h100_profiling/ a6000:/path/to/workdir/

# 2. On A6000, install deps
ssh a6000
cd /path/to/workdir/h100_profiling
pip install torch torchvision transformers diffusers accelerate safetensors

# 3. Run NCU profiling (4-6 hours, B=2 only)
cd a6000
./ncu_profile_all.sh

# 4. Transfer results back
# From local machine:
scp 'a6000:/path/to/workdir/h100_profiling/a6000/ncu_*.csv' ./a6000_results/
```

## Run on H100 (Paid - RunPod)

```bash
# 1. Launch H100 pod on RunPod
#    - H100 PCIe ($1.99/hr)
#    - 50GB container disk
#    - 6h auto-stop timer
#    - Template: RunPod PyTorch

# 2. Transfer scripts
scp -r ~/h100_profiling/ <runpod-ssh>:/workspace/

# 3. On H100, install deps
ssh <runpod-ssh>
cd /workspace/h100_profiling
pip install transformers diffusers accelerate safetensors

# 4. Test setup (5 minutes)
./test_setup.sh

# 5. Run full sweep (3-5 hours, all batches)
cd h100
./run_sweep_h100.sh

# 6. Download results
# From local machine:
scp '<runpod-ssh>:/workspace/h100_profiling/h100/*.json' ./h100_results/
scp '<runpod-ssh>:/workspace/h100_profiling/h100/*.txt' ./h100_results/

# 7. STOP THE POD!
ssh <runpod-ssh>
runpodctl stop pod $RUNPOD_POD_ID
```

## What You'll Get

### A6000 Output (Memory)
- `ncu_<model>_b2.csv` × 13 models
- Memory sector counts (DRAM, L2, L1)
- Energy metrics

### H100 Output (Performance)
- `trace_<model>_b<batch>.json` - Chrome traces
- `kernels_<model>_b<batch>.json` - Per-kernel stats
- `profile_<model>_b<batch>.txt` - Human-readable tables

## Time & Cost

| Phase | Time | Cost |
|-------|------|------|
| A6000 NCU | 4-6 hrs | $0 |
| H100 sweep | 3-5 hrs | $6-10 |
| **Total** | **7-11 hrs** | **$6-10** |

## 13 Model Configs

1. unet-sd (SD 2.1)
2. sdxl (SD XL)
3. dit-xl (DiT)
4. unet-3d (3D UNet)
5. llama-8b-1k
6. llama-8b-4k
7. llama-8b-16k
8. llama-8b-64k
9. resnet-50 (dense anchor)
10. cogvideox-16f-480p
11. cogvideox-49f-480p
12. cogvideox-81f-480p
13. cogvideox-49f-720p

## Troubleshooting

**OOM errors**: Normal! Script stops at max batch size for each model.

**Model access**: Run `huggingface-cli login` for Llama-3.

**See**: README.md for full details.
