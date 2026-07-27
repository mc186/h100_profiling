# H100 Profiling for HPCA Paper Resubmission

Complete profiling suite for hybrid-AI accelerator workload characterization.

## Overview

**Goal**: Capture per-layer performance and memory characteristics across 13 model configurations to support the "dense-vs-attention compute imbalance" claim.

**Two-phase approach**:
1. **A6000** (free): NCU memory profiling at B=2 for all models
2. **H100** (paid): PyTorch performance profiling at all batch sizes

## Model Configurations (13 total)

### Vision/Diffusion (4)
- `unet-sd`: Stable Diffusion 2.1 UNet
- `sdxl`: Stable Diffusion XL
- `dit-xl`: Diffusion Transformer XL
- `unet-3d`: 3D UNet for video

### Language (4)
- `llama-8b-1k`: Llama-8B @ 1024 seq length (prefill)
- `llama-8b-4k`: Llama-8B @ 4096 seq length
- `llama-8b-16k`: Llama-8B @ 16384 seq length
- `llama-8b-64k`: Llama-8B @ 65536 seq length

### Dense Anchor (1)
- `resnet-50`: Pure conv/dense (minimal attention)

### Video Generation (4)
- `cogvideox-16f-480p`: 16 frames @ 480×720 (less temporal)
- `cogvideox-49f-480p`: 49 frames @ 480×720 (baseline)
- `cogvideox-81f-480p`: 81 frames @ 480×720 (more temporal)
- `cogvideox-49f-720p`: 49 frames @ 720×1280 (higher spatial)

## Phase 1: A6000 NCU Profiling

**Machine**: Your free A6000  
**Runtime**: 4-6 hours  
**Batch size**: B=2 only (representative for spatial partitioning)  
**Cost**: $0  

### Setup

```bash
# Install dependencies
pip install torch torchvision transformers diffusers accelerate safetensors

# Copy scripts to A6000
scp -r h100_profiling/ a6000:/path/to/workdir/

# On A6000
cd /path/to/workdir/h100_profiling/a6000
```

### Run NCU Profiling

```bash
chmod +x ncu_profile_all.sh
./ncu_profile_all.sh
```

**This will**:
- Profile all 13 models at batch=2
- Capture: GPU time, SM utilization, DRAM/L2/L1 read/write sectors
- Generate: `ncu_<model>_b2.csv` files

**Expected output**:
```
ncu_unet-sd_b2.csv
ncu_sdxl_b2.csv
ncu_dit-xl_b2.csv
...
ncu_cogvideox-49f-720p_b2.csv
```

### NCU Metrics Collected

- `gpu__time_duration.sum`: Kernel execution time
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: Utilization (%)
- `dram__sectors_read.sum` / `dram__sectors_write.sum`: DRAM traffic
- `lts__t_sectors_read.sum` / `lts__t_sectors_write.sum`: L2 traffic
- `l1tex__t_sectors_read.sum` / `l1tex__t_sectors_write.sum`: L1 traffic

### Transfer Results

```bash
# From your local machine
scp a6000:/path/to/workdir/h100_profiling/a6000/ncu_*.csv ./a6000_results/
```

## Phase 2: H100 Performance Profiling

**Machine**: RunPod H100 PCIe  
**Runtime**: 3-5 hours  
**Batch sizes**: 1, 2, 4, 8, 16, 32, 64, 128 (stops on OOM)  
**Cost**: ~$6-10  

### Setup RunPod H100

1. **Launch H100 Pod**:
   - Go to RunPod → Deploy → GPU Pods
   - Select **H100 PCIe** ($1.99/hr)
   - Template: **RunPod PyTorch**
   - Container disk: **50 GB**
   - Docker command:
     ```bash
     bash -c "nohup sleep 6h; runpodctl stop pod $RUNPOD_POD_ID" &
     ```
   - Deploy

2. **SSH in**:
   ```bash
   ssh <runpod-ssh-command>
   ```

3. **Install dependencies**:
   ```bash
   pip install transformers diffusers accelerate safetensors
   ```

4. **Upload scripts**:
   ```bash
   # From your local machine
   scp -r h100_profiling/ <runpod-ssh>:/workspace/
   ```

### Run Performance Sweep

```bash
# On H100 pod
cd /workspace/h100_profiling/h100
chmod +x run_sweep_h100.sh
./run_sweep_h100.sh
```

**This will**:
- Profile all 13 models across all batch sizes
- Capture: Per-kernel runtime, operation class (attention/dense), utilization
- Generate: `trace_*.json`, `kernels_*.json`, `profile_*.txt` files

**Progress tracking**:
- Watch real-time: `tail -f sweep.log`
- Stops automatically on OOM for each model

### Download Results

```bash
# From your local machine
scp '<runpod-ssh>:/workspace/h100_profiling/h100/*.json' ./h100_results/
scp '<runpod-ssh>:/workspace/h100_profiling/h100/*.txt' ./h100_results/
```

### Stop H100 Pod

```bash
# Inside pod
runpodctl stop pod $RUNPOD_POD_ID

# Or from RunPod UI
# Pods → Your Pod → STOP
```

## Output Files

### A6000 NCU Results

```
ncu_<model>_b2.csv
```

**Format**: CSV with per-kernel rows  
**Columns**: kernel name, GPU time, utilization, DRAM/L2/L1 sectors

### H100 Performance Results

```
trace_<model>_b<batch>.json      # Chrome trace (visualize in chrome://tracing)
kernels_<model>_b<batch>.json    # Machine-readable per-kernel stats
profile_<model>_b<batch>.txt     # Human-readable table
```

**kernels JSON format**:
```json
{
  "model": "llama-8b-4k",
  "batch": 8,
  "kernels": [
    {
      "kernel_name": "matmul_kernel",
      "op_class": "dense",
      "cuda_time_us": 1234.5,
      "count": 10,
      "avg_time_us": 123.45
    },
    {
      "kernel_name": "attention_kernel",
      "op_class": "attention",
      "cuda_time_us": 567.8,
      "count": 5,
      "avg_time_us": 113.56
    }
  ]
}
```

## Post-Processing (Next Steps)

After collecting all data, you'll need to:

1. **Parse NCU CSVs** → extract per-layer memory metrics
2. **Parse kernels JSONs** → aggregate per-layer runtime/utilization
3. **Combine into final CSV**:
   ```csv
   model,batch,layer_index,op_class,runtime_ms,utilization_%,dram_read_sectors,dram_write_sectors,l2_read_sectors,l2_write_sectors,energy_pJ
   ```

4. **Validate**:
   - Total energy = power × runtime (catches unit errors)
   - Per-layer runtimes sum to end-to-end runtime

## Time & Cost Summary

| Phase | Machine | Time | Cost |
|-------|---------|------|------|
| NCU memory profiling | A6000 (free) | 4-6 hrs | $0 |
| Performance sweep | H100 RunPod | 3-5 hrs | $6-10 |
| **Total** | | **7-11 hrs** | **$6-10** |

## Troubleshooting

### OOM Errors
- Normal! Script automatically stops batch sweep when OOM occurs
- Each model will succeed up to its maximum batch size

### Model Loading Errors
- Check you have Hugging Face access to gated models (Llama-3)
- Run `huggingface-cli login` if needed

### NCU Slowness
- NCU profiling is 10-100× slower than normal inference
- Be patient, each model takes 20-40 minutes

### Dependencies
```bash
# Full install command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers accelerate safetensors huggingface-hub
pip install imageio-ffmpeg opencv-python  # For video models
```

## Contact

For questions about this profiling suite, refer to your HPCA paper draft or contact the team.

---

**Last updated**: 2026-07-23
