#!/bin/bash
##############################################################################
# Quick Setup Test
#
# Purpose: Verify dependencies and test one model before running full sweep
# Runtime: ~5-10 minutes
##############################################################################

echo "============================================================"
echo "Testing H100 Profiling Setup"
echo "============================================================"
echo ""

# Check Python
echo "Checking Python..."
python --version || { echo "✗ Python not found"; exit 1; }
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi || { echo "✗ nvidia-smi not found"; exit 1; }
echo ""

# Check PyTorch + CUDA
echo "Checking PyTorch + CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || { echo "✗ PyTorch/CUDA issue"; exit 1; }
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import transformers; import diffusers; print('✓ transformers + diffusers installed')" || { echo "✗ Missing transformers or diffusers"; exit 1; }
echo ""

# Test one simple model (ResNet-50)
echo "Testing ResNet-50 (simple, no model download)..."
if python run_model.py --model resnet-50 --batch 1; then
    echo "✓ ResNet-50 works!"
else
    echo "✗ ResNet-50 failed"
    exit 1
fi
echo ""

# Test PyTorch profiler
echo "Testing PyTorch profiler..."
cd h100
if python profile_h100.py --model resnet-50 --batch 1; then
    echo "✓ PyTorch profiler works!"
    echo "  Generated files:"
    ls -lh trace_resnet-50_b1.json kernels_resnet-50_b1.json profile_resnet-50_b1.txt
else
    echo "✗ Profiler failed"
    exit 1
fi
cd ..
echo ""

echo "============================================================"
echo "✓ Setup test PASSED!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review generated test files in h100/"
echo "  2. Run full sweep: cd h100 && ./run_sweep_h100.sh"
echo ""
echo "Estimated costs:"
echo "  - Full H100 sweep: ~$6-10 (3-5 hours)"
echo "  - Remember to STOP the pod when done!"
echo "============================================================"
