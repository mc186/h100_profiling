#!/bin/bash
##############################################################################
# H100 Performance Sweep (All Batches)
#
# Purpose: Profile all models across all batch sizes using PyTorch profiler
# Runtime: ~3-5 hours for all models × batches
# Output: trace_*.json, kernels_*.json, profile_*.txt
##############################################################################

set -e  # Exit on error (will stop on OOM)

# All 13 model configurations
MODELS=(
    "unet-sd"
    "sdxl"
    "dit-xl"
    "unet-3d"
    "llama-8b-1k"
    "llama-8b-4k"
    "llama-8b-16k"
    "llama-8b-64k"
    "resnet-50"
    "cogvideox-16f-480p"
    "cogvideox-49f-480p"
    "cogvideox-81f-480p"
    "cogvideox-49f-720p"
)

# Batch sizes to sweep
BATCHES=(1 2 4 8 16 32 64 128)

echo "============================================================"
echo "H100 Performance Sweep"
echo "============================================================"
echo "Models: ${#MODELS[@]}"
echo "Batches per model: ${#BATCHES[@]}"
echo "Total runs (max): $((${#MODELS[@]} * ${#BATCHES[@]}))"
echo "Expected runtime: 3-5 hours"
echo ""
echo "Starting at: $(date)"
echo "============================================================"
echo ""

START_TIME=$(date +%s)
TOTAL_RUNS=0
SUCCEEDED=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model: ${MODEL}"
    echo "========================================"

    for BATCH in "${BATCHES[@]}"; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        echo "  [${TOTAL_RUNS}] Batch=${BATCH}... $(date +%H:%M:%S)"

        # Try to run, catch OOM
        if python profile_h100.py --model ${MODEL} --batch ${BATCH} 2>&1 | tee -a sweep.log; then
            echo "  ✓ Success"
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  ✗ FAILED (likely OOM) - stopping batch sweep for ${MODEL}"
            FAILED=$((FAILED + 1))
            break  # Stop trying larger batches for this model
        fi

        # Clean up GPU memory
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        sleep 3
    done

    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "============================================================"
echo "Sweep Complete!"
echo "============================================================"
echo "Succeeded: ${SUCCEEDED}"
echo "Failed (OOM): ${FAILED}"
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo "Finished at: $(date)"
echo ""
echo "Generated files:"
echo "  Traces: $(ls trace_*.json 2>/dev/null | wc -l) files"
echo "  Kernels: $(ls kernels_*.json 2>/dev/null | wc -l) files"
echo "  Profiles: $(ls profile_*.txt 2>/dev/null | wc -l) files"
echo ""
echo "Download results:"
echo "  scp '<runpod-ssh>:/workspace/*.json' ./h100_results/"
echo "  scp '<runpod-ssh>:/workspace/*.txt' ./h100_results/"
echo ""
echo "IMPORTANT: Stop the H100 pod to avoid charges!"
echo "  runpodctl stop pod \$RUNPOD_POD_ID"
echo "============================================================"
