#!/bin/bash
##############################################################################
# A6000 NCU Memory Profiling (B=2 only)
#
# Purpose: Capture per-layer memory sector counts and energy for all models
# at batch size 2 (representative for spatial partitioning analysis)
#
# Runtime: ~4-6 hours for all 13 models
# Output: ncu_<model>_b2.csv files
##############################################################################

set -e  # Exit on error

# All models to profile at B=2 for memory characterization
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

BATCH=2

echo "============================================================"
echo "A6000 NCU Memory Profiling"
echo "============================================================"
echo "Models: ${#MODELS[@]}"
echo "Batch size: ${BATCH}"
echo "Expected runtime: 4-6 hours"
echo ""
echo "NCU Metrics Collected:"
echo "  - GPU time duration"
echo "  - SM throughput (utilization)"
echo "  - DRAM read/write sectors"
echo "  - L2 (LTS) read/write sectors"
echo "  - L1 read/write sectors"
echo ""
echo "Starting at: $(date)"
echo "============================================================"
echo ""

START_TIME=$(date +%s)
SUCCEEDED=0
FAILED=0

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model: ${MODEL}"
    echo "Batch: ${BATCH}"
    echo "Time: $(date +%H:%M:%S)"
    echo "========================================"

    OUTPUT_FILE="ncu_${MODEL}_b${BATCH}.csv"

    # NCU command for memory sectors + energy
    if ncu --metrics \
        gpu__time_duration.sum,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__sectors_read.sum,\
        dram__sectors_write.sum,\
        lts__t_sectors_read.sum,\
        lts__t_sectors_write.sum,\
        l1tex__t_sectors_read.sum,\
        l1tex__t_sectors_write.sum \
        --kernel-name-base demangled \
        --target-processes all \
        --csv \
        --log-file "${OUTPUT_FILE}" \
        python ../run_model.py --model ${MODEL} --batch ${BATCH}; then

        echo "✓ SUCCESS: ${OUTPUT_FILE}"
        SUCCEEDED=$((SUCCEEDED + 1))

        # Show file size
        SIZE=$(ls -lh "${OUTPUT_FILE}" | awk '{print $5}')
        echo "  File size: ${SIZE}"
    else
        echo "✗ FAILED: ${MODEL} (likely OOM or model loading error)"
        FAILED=$((FAILED + 1))
    fi

    # Clean GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5

    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "============================================================"
echo "NCU Profiling Complete!"
echo "============================================================"
echo "Succeeded: ${SUCCEEDED}/${#MODELS[@]}"
echo "Failed: ${FAILED}/${#MODELS[@]}"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Finished at: $(date)"
echo ""
echo "Generated files:"
ls -lh ncu_*.csv 2>/dev/null || echo "  (none)"
echo ""
echo "Next step: Transfer these CSV files to your analysis machine"
echo "  scp a6000:/path/to/ncu_*.csv ./a6000_results/"
echo "============================================================"
