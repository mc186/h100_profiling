# Session Context — H100 Profiling (HPCA resubmission)

Last updated: 2026-07-27. This file is a handoff note for a fresh Claude Code session.

## Goal
Profile 13 model configs on an H100 (RunPod) to characterize the dense-vs-attention
compute imbalance for a hybrid-AI accelerator paper. Per-model runtime + attention/dense
split via PyTorch profiler. (Deep NCU memory profiling at B=2 is a SEPARATE phase planned
for a free A6000 — not started yet.)

## Where things run
- **Local machine**: `atletx8-reg172` (this box, RHEL8). Repo lives at `/home/manschou/h100_profiling`,
  pushed to https://github.com/mc186/h100_profiling.
- **H100**: RunPod pod, SSH `1cxrns12f92ngv-64410ca6@ssh.runpod.io`. Work happens in Jupyter
  at `/workspace/h100_profiling` on the pod. NOTE: direct SSH from the local box fails
  (no PTY / publickey denied) — drive the pod via Jupyter or the RunPod MCP server.
- RunPod pod runs CUDA 13.0 driver / PyTorch cu128. A prior pod had a CUDA-busy init failure
  and was scrapped; the current one works.

## RunPod MCP server (just set up)
- Node 20 installed via nvm and set as default (system node is 14, too old).
- Wrapper: `~/.local/bin/runpod-mcp-wrapper.sh` (pins node 20, runs `npx @runpod/mcp-server`).
- Registered with `claude mcp add runpod` using the user's RunPod API key (stored in
  `~/.claude.json`). Health check passed (`claude mcp list` → ✓ Connected).
- The `runpod` MCP tools load only in a session started AFTER registration — hence this handoff.
- Caveat: the MCP server manages pod LIFECYCLE via the RunPod API. It may NOT provide an
  arbitrary shell inside the pod. Confirm the actual tool set first; if no in-pod exec,
  fall back to giving the user one-line Jupyter cells to run.

## Models (13) and status
Working (profiled or expected to work): resnet-50, unet-sd (SD v1.5), sdxl, unet-3d,
cogvideox-16f-480p / 49f-480p / 81f-480p / 49f-720p.
Still failing — the two to fix:
- **llama-8b-1k/4k/16k/64k**: last error "Asking to pad but the tokenizer does not have a
  pad token." run_model.py was updated to use `meta-llama/Llama-2-7b-hf` and set
  `tokenizer.pad_token = tokenizer.eos_token`, but the user reports it still doesn't run —
  ROOT CAUSE NOT YET CONFIRMED. Likely the pod hadn't `git pull`ed the fix, OR Llama-2 is
  gated and needs `huggingface-cli login`. NEED to see the actual traceback from the pod.
- **dit-xl**: a string of failures — VAE `diffusion_pytorch_model.safetensors` not found,
  then `sd-vae-ft-msa` not a local folder, then disk-quota-exceeded, then corrupted cache,
  then `DiTPipeline.__call__() got an unexpected keyword`. run_model.py currently loads DiT
  with SD-v1.5 VAE. User INSISTS on fixing DiT (do NOT suggest skipping it again).

## Known environment gotchas
- **Disk quota**: the pod fills up fast. HuggingFace cache at `/workspace/.cache/huggingface/hub`
  must be cleared between models. The notebook's `cleanup_model()` does this.
- **PyTorch profiler API**: newer torch renamed `cuda_time_total` → use `device_time_total`
  fallback (already handled in the OneByOne notebook's `profile_model`).
- torchvision: `resnet50(pretrained=True)` deprecated → use `weights="IMAGENET1K_V1"` (fixed).
- SD 2.1 was deprecated/removed from HF → switched to `runwayml/stable-diffusion-v1-5` (fixed).

## Results
User has already DOWNLOADED partial results (a `summary.json` with ~53 rows plus
`trace_*.json` / `kernels_*.json` under `/workspace/results`). Do NOT force a full re-run —
only fill the gaps (Llama + DiT). Output schema per row: model, batch, status, total_time_ms,
attention_time_ms, dense_time_ms, attention_pct, dense_pct.

## Repo layout
- `run_model.py` — MODEL_RUNNERS dispatch, all 13 configs. The source of truth for model loading.
- `H100_Profiling_OneByOne.ipynb` — the notebook actually being used (per-model cells,
  cleanup between models, profiler with version-safe timing). Preferred over the bash scripts.
- `H100_Profiling.ipynb` — older all-in-one version.
- `h100/`, `a6000/` — bash-script variants (SSH-based, largely superseded by the notebook).

## Immediate next step
1. Confirm the `runpod` MCP tools are available; list the user's pods.
2. Get the REAL Llama traceback from the pod (gated-model login vs. stale code vs. pad token).
3. Fix DiT-XL for real — needs the actual current error, reproduced, not guessed.
Interactive-first: reproduce on the pod BEFORE editing run_model.py and pushing.
