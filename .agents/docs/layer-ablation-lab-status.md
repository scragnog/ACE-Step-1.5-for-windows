---
description: Current status and context for the Layer Ablation Lab feature and related fixes
---

# Layer Ablation Lab — Status & Continuation Guide

## Goal

The Layer Ablation Lab is a **developer tool** in the ACE-Step UI that helps understand what each of the 24 DiT transformer layers contributes to the audio output when a LoRA/LoKR adapter is loaded. The workflow is:

1. Load an adapter, generate a reference track with a fixed seed
2. Zero specific layer scales (set their adapter contribution to 0), regenerate with the same seed
3. Audio Diff the two outputs — the RMS energy shows how much that layer mattered
4. Repeat to build a map of which layers control vocals, instrumentation, timbre, etc.

## Architecture

### Frontend (`ace-step-ui/`)

- **`LayerAblationPanel.tsx`** — the UI panel with layer toggle buttons, zero/reset buttons, and audio diff
- **`CreatePanel.tsx`** — parent component that owns `adapterSlots` state and provides callbacks:
  - `handleSlotLayerScaleChange(slot, layer, scale)` — debounced (500ms), updates local state immediately for visual feedback, then sends ALL current layer scales as a batch API call
  - `handleBulkLayerScalesChange(slot, layerScales)` — used by zero/reset buttons, single batch call
- **`AdaptersAccordion.tsx`** — renders the per-layer scale sliders

### Backend (`acestep/`)

- **`api_server.py`** — endpoints:
  - `POST /v1/lora/slot-layer-scale` — single layer (used by individual slider, now deprecated in favor of batch)
  - `POST /v1/lora/slot-layer-scales` — batch (receives full `{layer: scale}` dict, REPLACES entire dict)
  - `POST /api/lora/slot-layer-scale` and `/api/lora/slot-layer-scales` — aliased versions
- **`advanced_adapter_mixin.py`** — core merge logic:
  - `set_slot_layer_scales()` — stores `layer_scales` dict in `_adapter_slots[slot]`, triggers merge
  - `set_slot_layer_scale()` — single layer, merges into existing dict
  - `_apply_merged_weights_with_groups()` — computes `base + Σ(slot_scale × group_scale × layer_scale × delta)` for each weight key
  - `_apply_merged_weights()` — older version, same logic
  - `_extract_layer_index(key)` — parses `layers.N` from weight key names, returns int or None

## Bugs Fixed in This Session

### 1. CUDA Graph Capture Failure (Root Cause of Adapter Load Failure)
**Files:** `llm_inference.py`, `nanovllm/layers/attention.py`

- `flash_attn` is NOT installed → nanovllm uses SDPA fallback
- SDPA decode uses `.item()` (line 267) → illegal during CUDA graph capture
- Failed `capture_cudagraph()` poisons entire CUDA context → all subsequent CUDA ops fail
- **Fix:** Force `enforce_eager=True` when `flash_attn` is missing (skip graph capture)
- Also wrote dual-path SDPA decode: efficient `.item()` path for normal inference, batched path for graph capture (if ever needed)
- The SDPA fix is in both `acestep/third_parts/nano-vllm/` AND `.venv/Lib/site-packages/nanovllm/` (copied manually since pip install had issues)

### 2. 40GB Shared Memory Spike
**Files:** `nanovllm/layers/attention.py`

- First attempt at SDPA fix used batched gather for ALL calls → massive intermediate tensors
- Fixed by using `torch.cuda.is_current_stream_capturing()` to only use batched path during graph capture
- Then further fixed by adding `enforce_eager=True` to skip graph capture entirely without `flash_attn`

### 3. Ablation Panel Buttons Not Updating Sliders
**Files:** `LayerAblationPanel.tsx`, `CreatePanel.tsx`

- Panel called `generateApi.setSlotLayerScale()` directly, bypassing CreatePanel's state
- Fixed by adding `onLayerScaleChange` and `onBulkLayerScalesChange` callback props

### 4. Zero/Reset Firing 24 Sequential API Calls (72 seconds)
**Files:** `LayerAblationPanel.tsx`, `CreatePanel.tsx`

- Each call triggered a full weight merge (~3s each)
- Fixed by using batch `setSlotLayerScales` API (single call)

### 5. Corrupt Audio with All Layers Zeroed
**Files:** `advanced_adapter_mixin.py`

- Non-layer keys (norms, embeddings) had `layer_idx = None` → defaulted to `l_scale = 1.0`
- Full adapter delta applied even when all 24 layer scales were 0
- **Fix:** Non-layer keys now use average of all set layer scales
- Fixed in BOTH `_apply_merged_weights` and `_apply_merged_weights_with_groups`

### 6. Layer Slider Spam (Dragging Fires Continuous API Calls)
**Files:** `CreatePanel.tsx`

- Slider `onChange` fires per-pixel during drag, each triggering a weight merge
- **Fix:** 500ms debounce — state updates immediately (visual), API call fires once after user stops

### 7. Debounced Batch Wiping Other Layer Scales
**Files:** `CreatePanel.tsx`

- Debounce sent only changed layers → batch endpoint REPLACED entire dict → wiped other scales
- **Fix:** Debounce now reads ALL current layer scales from React state at flush time
- **STATUS: Just committed, needs testing by user**

## Current State / What Needs Testing

### Immediate Testing Needed
1. **Restart app** and verify layer scale changes work correctly:
   - Zero all 24 layers → audio should match base model (no adapter)
   - Set layer 0 to 2.0 while others stay at 0 → should hear only layer 0's contribution
   - Drag a slider → should see ONE API call after releasing (not continuous spam)
   - Reset All to 1.0 → should restore full adapter audio

### Known Issues / Future Work
- **`flash_attn` not installed** — SDPA is slower. Installing `flash_attn` on Windows would enable CUDA graph capture and faster LLM inference. This is a follow-up task.
- **CPU-move workaround** in `_extract_adapter_delta()` — model moves to CPU for LoKR/PEFT injection to avoid CUDA graph state issues. Works but is slow. Would be unnecessary if `enforce_eager` fully resolves the CUDA context issues.
- **Audio Diff endpoint** (`/api/audio-diff`) — exists but hasn't been tested in this session

## Key Files

| File | Role |
|------|------|
| `acestep/llm_inference.py` | LLM init, `enforce_eager` flash_attn detection |
| `acestep/core/generation/handler/lora/advanced_adapter_mixin.py` | Adapter loading, delta extraction, weight merging |
| `acestep/third_parts/nano-vllm/nanovllm/layers/attention.py` | SDPA/flash_attn attention (CUDA graph compat) |
| `ace-step-ui/components/CreatePanel.tsx` | Layer scale handlers, debounce logic |
| `ace-step-ui/components/accordions/LayerAblationPanel.tsx` | Ablation UI, zero/reset buttons |
| `ace-step-ui/components/accordions/AdaptersAccordion.tsx` | Per-layer scale sliders |
| `ace-step-ui/services/api.ts` | API client (`setSlotLayerScale`, `setSlotLayerScales`) |

## Git State

All changes are committed to the `qinglong` branch. Recent commits (newest first):
- `chore: update ace-step-ui submodule (fix debounce layer scales)`
- `fix: non-layer keys (norms, embeddings) now respect layer scales`
- `fix: force enforce_eager when flash_attn is missing`
- `fix: dual-path SDPA decode — efficient .item() for inference, batched for graph capture`
- `fix: make SDPA decode CUDA-graph-compatible (no .item() calls)`

The `ace-step-ui` submodule also has its own commits on a detached HEAD.
