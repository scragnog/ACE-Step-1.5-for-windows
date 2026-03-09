# Temporal Adapter Switching & Per-Layer Adapter Scales

## Goal

Two complementary features that enable multi-singer songs and fine-grained adapter control:

1. **Per-Layer Adapter Scales** — Control which transformer layers each adapter affects, enabling isolation of vocal vs. instrumentation characteristics
2. **Temporal Weight-Space Interpolation** — Smoothly transition adapter weights during diffusion, enabling verse/chorus singer switching without jarring artifacts

## Background

### Current Architecture

The advanced adapter system uses weight-space merging:
```
decoder_weights = base + Σ(slot_scale × group_scale × delta[key])
```

- **4 adapter slots**, each with independent `scale` and `group_scales` (self_attn, cross_attn, mlp)
- Weights are merged ONCE before generation via `_apply_merged_weights_with_groups()`
- The merged decoder runs identically for all diffusion steps
- DiT has **24 transformer layers** (`layers.0` through `layers.23`)
- Delta dict keys are fully qualified: `layers.7.attn.qkv.weight`, `layers.12.cross_attn.k_proj.weight`, etc.

### Key Files

| File | Role |
|------|------|
| [advanced_adapter_mixin.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/advanced_adapter_mixin.py) | Weight-space merging, slot management, group scales |
| [modeling_acestep_v15_base.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/models/base/modeling_acestep_v15_base.py) | `generate_audio()` — the 8-step denoising loop (line 2049) |
| [service_generate.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/service_generate.py) | Top-level generation orchestration |
| [controls_test.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/controls_test.py) | Existing adapter control tests |
| [lifecycle_test.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/lifecycle_test.py) | Existing adapter lifecycle tests |

---

## Feature 1: Per-Layer Adapter Scales

### What Changes

Extend the merge formula from:
```
base + Σ(slot_scale × group_scale × delta[key])
```
to:
```
base + Σ(slot_scale × group_scale × layer_scale × delta[key])
```

Where `layer_scale` is looked up from a per-slot, per-layer scale dict.

### Proposed Changes

---

#### [MODIFY] [advanced_adapter_mixin.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/advanced_adapter_mixin.py)

**1. Add `_extract_layer_index()` helper** (new function, ~10 lines)

Extract the layer index from a delta key name. Example: `layers.7.attn.qkv.weight` → `7`.

```python
def _extract_layer_index(key: str) -> int | None:
    """Extract transformer layer index from a weight key, e.g. 'layers.7.attn...' -> 7."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
```

**2. Add `layer_scales` to slot data structure** (modify `load_lora_slot`)

When a slot is created (line 412-419), add:
```python
"layer_scales": {},  # empty = all layers at 1.0
```

**3. Extend `_apply_merged_weights_with_groups`** (modify existing function)

Add layer_scale lookup in the merge loop (line 329-337):
```python
for k in self._base_decoder:
    base_val = self._base_decoder[k]
    if k in all_keys:
        group = _determine_group(k)
        layer_idx = _extract_layer_index(k)
        combined = base_val.float()
        for s in active_slots.values():
            if k in s["delta"]:
                g_scale = s.get("group_scales", {}).get(group, 1.0)
                l_scale = s.get("layer_scales", {}).get(layer_idx, 1.0) if layer_idx is not None else 1.0
                combined = combined + s["scale"] * g_scale * l_scale * s["delta"][k]
        merged[k] = combined.to(dtype=base_val.dtype)
```

**4. Add `set_slot_layer_scales()` public API** (new function)

```python
def set_slot_layer_scales(self, slot: int, layer_scales: Dict[int, float]) -> str:
    """Set per-layer LoRA scales for a specific adapter slot.
    
    Args:
        slot: Slot ID
        layer_scales: Dict mapping layer index (0-23) to scale (0.0-2.0).
                       Unlisted layers default to 1.0.
    """
```

**5. Add `set_slot_layer_scale()` single-layer convenience** (new function)

```python
def set_slot_layer_scale(self, slot: int, layer: int, scale: float) -> str:
    """Set scale for a single layer on a specific adapter slot."""
```

**6. Update `get_advanced_lora_status`** to include layer_scales in slot info.

---

#### [MODIFY] [handler.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/handler.py)

Expose new `set_slot_layer_scales` and `set_slot_layer_scale` methods on the handler.

---

#### UI Changes (Phase 2 — not in initial implementation)

> [!NOTE]
> The UI for per-layer scales is an advanced feature. Initial implementation exposes the API only, with UI controls added later once we understand which layers matter through experimentation.

---

## Feature 2: Temporal Weight-Space Interpolation

### Concept

Instead of merging adapter weights once before generation, re-merge at each diffusion step with time-varying slot scales. A "schedule" defines how each slot's effective scale changes over the song's temporal dimension.

### How Temporal Mapping Works

The diffusion loop iterates over **denoising steps**, not song timesteps. But the underlying latent tensor has a temporal dimension — each frame corresponds to a position in the song. The adapter schedule maps **song position** (0.0–1.0) to **slot scale** per adapter.

Example schedule for "verse = Adapter A, chorus = Adapter B":
```python
adapter_schedule = {
    0: [  # Slot 0 (Singer A)
        {"start": 0.0, "end": 0.45, "scale": 1.0},   # Full for verse
        {"start": 0.45, "end": 0.55, "scale": "fade"},  # Crossfade
        {"start": 0.55, "end": 1.0, "scale": 0.0},    # Off for chorus
    ],
    1: [  # Slot 1 (Singer B)
        {"start": 0.0, "end": 0.45, "scale": 0.0},    # Off for verse
        {"start": 0.45, "end": 0.55, "scale": "fade"},  # Crossfade
        {"start": 0.55, "end": 1.0, "scale": 1.0},    # Full for chorus
    ],
}
```

> [!IMPORTANT]
> **Key design decision**: Should we re-merge the full decoder state_dict at each step (expensive but simple), or modify the decoder forward pass to apply per-frame per-layer deltas inline (complex but efficient)?
>
> **Recommendation**: Start with per-step re-merge. The DiT phase is only ~7% of total generation time. Even doubling it (re-merging 8 times for 8 steps) adds only ~7% to total time. This keeps the code simple and avoids modifying the model architecture.

### Proposed Changes

---

#### [NEW] [temporal_adapter_schedule.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/temporal_adapter_schedule.py)

New module defining the schedule data structure and interpolation logic:

```python
@dataclass
class AdapterSegment:
    start: float      # Song position 0.0-1.0
    end: float        # Song position 0.0-1.0
    scale: float      # Adapter scale for this segment
    fade_in: float    # Crossfade duration at start (seconds)
    fade_out: float   # Crossfade duration at end (seconds)

@dataclass
class TemporalAdapterSchedule:
    slot_segments: Dict[int, List[AdapterSegment]]
    
    def get_effective_scales(self, song_position: float) -> Dict[int, float]:
        """Get the interpolated scale for each slot at a given song position."""
```

---

#### [MODIFY] [advanced_adapter_mixin.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/lora/advanced_adapter_mixin.py)

**1. Add `_apply_merged_weights_temporal()` method**

Like `_apply_merged_weights_with_groups` but takes a `schedule_scales: Dict[int, float]` parameter that overrides per-slot scales for that specific step:

```python
def _apply_merged_weights_temporal(self, schedule_scales: Dict[int, float]) -> None:
    """Merge with per-slot scale overrides for temporal interpolation."""
```

**2. Add `set_temporal_schedule()` public API**

```python
def set_temporal_schedule(self, schedule: Optional[TemporalAdapterSchedule]) -> str:
    """Set or clear the temporal adapter schedule for the next generation."""
```

---

#### [MODIFY] [service_generate_execute.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/core/generation/handler/service_generate_execute.py)

OR

#### [MODIFY] [modeling_acestep_v15_base.py](file:///D:/Ace-Step-Latest/ACE-Step-1.5-for-windows/acestep/models/base/modeling_acestep_v15_base.py)

> [!WARNING]
> **Design decision needed**: The denoising loop lives in `generate_audio()` (the monkeypatched model method). Injecting the temporal re-merge here requires either:
> 
> **(A)** Passing a callback from the handler into `generate_audio()` that re-merges weights before each step. Clean separation but requires modifying the `generate_audio` signature.
> 
> **(B)** Moving the per-step callback into `service_generate_execute.py` by wrapping the `generate_audio` call with a step-aware hook system. More invasive but keeps model code clean.
> 
> **(C)** Using PyTorch forward hooks on the decoder that intercept each step and re-merge weights. Elegant but fragile.
> 
> **Recommendation**: **(A)** — Add an optional `on_step_callback` parameter to `generate_audio()`. The handler passes a lambda that calls `_apply_merged_weights_temporal()` with the schedule for the current step. This is the cleanest approach.

Add to the denoising loop (around line 2049):
```python
for step_idx, (t_curr, t_prev) in enumerate(iterator):
    # Temporal adapter re-merge (if schedule is set)
    if on_step_callback is not None:
        on_step_callback(step_idx=step_idx, t_curr=t_curr, total_steps=infer_steps)
    
    # ... existing diffusion step code ...
```

---

## Implementation Order

> [!TIP]
> Feature 1 (per-layer scales) is a prerequisite for the best use of Feature 2. Implement in this order:

### Phase 1: Per-Layer Scales (Backend Only)
- [ ] Add `_extract_layer_index()` helper
- [ ] Add `layer_scales` to slot data structure
- [ ] Extend `_apply_merged_weights_with_groups` with layer scale lookup
- [ ] Add `set_slot_layer_scales()` and `set_slot_layer_scale()` APIs
- [ ] Update `get_advanced_lora_status` to return layer_scales
- [ ] Expose on handler
- [ ] Unit tests

### Phase 2: Temporal Schedule (Backend Only)
- [ ] Create `temporal_adapter_schedule.py` with schedule data structures
- [ ] Add `_apply_merged_weights_temporal()` to adapter mixin
- [ ] Add `on_step_callback` parameter to `generate_audio()`
- [ ] Wire callback through `service_generate_execute.py`
- [ ] Add `set_temporal_schedule()` API
- [ ] Unit tests

### Phase 3: API Endpoints
- [ ] Add REST endpoints for per-layer scales
- [ ] Add REST endpoint for temporal schedule
- [ ] Integration tests

### Phase 4: UI
- [ ] Per-layer scale controls in Advanced Adapter panel
- [ ] Temporal schedule builder (visual timeline with adapter assignment)
- [ ] Preview/feedback mechanism

---

## Verification Plan

### Automated Tests

**Run existing tests** to ensure no regressions:
```powershell
D:\Ace-Step-Latest\ACE-Step-1.5-for-windows\.venv\Scripts\python.exe -m pytest acestep/core/generation/handler/lora/controls_test.py -v
D:\Ace-Step-Latest\ACE-Step-1.5-for-windows\.venv\Scripts\python.exe -m pytest acestep/core/generation/handler/lora/lifecycle_test.py -v
```

**New unit tests** for per-layer scales:
- `test_extract_layer_index()` — verify parsing of layer index from key names
- `test_layer_scale_zero_disables_layer()` — verify setting a layer to 0 excludes it from merge
- `test_layer_scale_default_is_one()` — verify unlisted layers get scale 1.0
- `test_slot_layer_scales_independent()` — verify different slots can have different layer scales
- `test_layer_scales_combined_with_group_scales()` — verify multiplicative combination

**New unit tests** for temporal schedule:
- `test_schedule_constant_segment()` — verify constant scale across a segment
- `test_schedule_crossfade()` — verify smooth interpolation during fade
- `test_schedule_edge_cases()` — verify boundary positions (0.0, 1.0)
- `test_temporal_merge_called_per_step()` — verify callback fires each step

### Manual Verification

User-driven testing (requires loaded adapters and full generation):

1. **Per-Layer Isolation Test**: Load two adapters. Set Adapter A: layer 12 scale=0. Set Adapter B: all layers except 12 scale=0. Generate and compare against:
   - Adapter A alone at full scale
   - Adapter B alone at full scale
   - Verify the output is different from either alone

2. **Temporal Switching Test**: Load two vocalist adapters. Set a schedule with Adapter A for first half, Adapter B for second half, 2-second crossfade. Generate a song and listen for:
   - Clean transition between vocalists
   - Consistent instrumentation through the transition
   - No audio artifacts at the crossfade boundary

> [!IMPORTANT]
> The specific layers that correspond to vocals vs. instrumentation in ACE-Step v1.5 are **unknown** and need experimental discovery. Once per-layer controls are implemented, a systematic test should be done: zero out one layer at a time and listen to what changes. This could be documented as a knowledge artifact for future reference.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Per-step re-merge too slow | +14% generation time worst case | Profile first; DiT is only 7% of total time |
| Layer isolation doesn't cleanly separate vocals/instruments | Feature less useful | Start with group-scale isolation; layer isolation is additive |
| Crossfade artifacts at transition boundaries | Audible glitches | Tune crossfade duration; test with 1-4 second fades |
| torch.compile invalidated by weight changes | Compilation cache thrash | May need to disable compile when temporal schedule is active |
| Memory: holding multiple decoder copies | VRAM pressure | Deltas are stored in FP32 on CPU, only merged copy goes to GPU |

---

## Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Per-Layer Scales | 1-2 sessions | Low — straightforward extension |
| Phase 2: Temporal Schedule | 2-3 sessions | Medium — touches diffusion loop |
| Phase 3: API Endpoints | 1 session | Low |
| Phase 4: UI | 2-3 sessions | Medium — timeline builder UX |
| **Total** | **~6-9 sessions** | |
