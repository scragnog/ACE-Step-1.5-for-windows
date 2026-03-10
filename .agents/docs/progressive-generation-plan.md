# Progressive Generation — Sequential Overlapping Chunks

> **Status:** Draft — not yet implemented  
> **Created:** 2026-03-10  
> **Approach:** Generate audio in ~30s chunks with overlap, allowing playback to begin before the full track is finished.

---

## Goal

Add an **opt-in "Progressive Generation"** mode that lets the user start listening to generated audio after the first ~30 seconds are ready, while subsequent chunks continue generating in the background. The existing full-sequence generation pipeline remains the default and is never affected.

---

## Why It's Non-Trivial

ACE-Step's DiT model uses **bidirectional (non-causal) self-attention** — every latent position attends to every other position. The diffusion loop denoises the **entire** latent sequence simultaneously across N steps. Unlike autoregressive models, you can't extract partial output mid-generation.

However, the model **natively handles variable-length sequences** (10s–240s), and the existing `chunk_masks` / repainting mechanism provides infrastructure for "locking" previously-generated latents while denoising new ones.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              Progressive Generation             │
│                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Chunk 1  │   │ Chunk 2  │   │ Chunk 3  │    │
│  │  ~30s    │──▶│  ~30s    │──▶│  ~30s    │    │
│  │ + 5s pad │   │ 5s ovlap │   │ 5s ovlap │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │ VAE          │ VAE          │ VAE       │
│       ▼ decode       ▼ decode       ▼ decode    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Audio 1  │   │ Audio 2  │   │ Audio 3  │    │
│  │ (play    │   │ (crossfade│   │ (crossfade│   │
│  │  immediately) │ & append)│   │ & append)│    │
│  └──────────┘   └──────────┘   └──────────┘    │
└─────────────────────────────────────────────────┘
```

---

## Detailed Design

### 1. Chunk Planner

Splits the target duration into chunks before generation begins.

**Input:** total duration (e.g. 180s), chunk size (default 30s), overlap (default 5s)  
**Output:** list of `ChunkSpec` objects:

```python
@dataclass
class ChunkSpec:
    chunk_index: int
    start_sec: float       # absolute start in full track
    end_sec: float         # absolute end in full track
    overlap_sec: float     # overlap with previous chunk
    lyrics_slice: str      # lyrics for this time range
    is_first: bool
    is_last: bool
```

**Example for a 90s track (30s chunks, 5s overlap):**

| Chunk | Latent range (25Hz) | Absolute time | Overlap |
|-------|-------------------|---------------|---------|
| 0     | [0, 750]          | 0s – 30s      | None    |
| 1     | [625, 1375]       | 25s – 55s     | 5s from chunk 0 |
| 2     | [1250, 2000]      | 50s – 80s     | 5s from chunk 1 |
| 3     | [1875, 2250]      | 75s – 90s     | 5s from chunk 2 |

### 2. Per-Chunk Diffusion

Each chunk runs a **full diffusion process** independently (same solver, same steps, same guidance). The key differences from normal generation:

- **Chunk 0:** Standard generation at the chunk's duration. No special handling.
- **Chunks 1–N:** 
  - The overlap zone contains **frozen latents** from the previous chunk's final output
  - `chunk_masks` are set so the overlap zone is masked (diffusion only denoises the new portion)
  - The overlap latents provide bidirectional context for the new portion
  - Alternatively, use the repainting mechanism: `repainting_start` / `repainting_end` set to the overlap zone

**Conditioning per chunk:**
- **Text/genre conditioning:** Same for all chunks (global)
- **Lyrics:** Sliced to match this chunk's time range (requires lyric timestamp alignment)
- **Timbre:** Same for all chunks (global)
- **Seed:** Derived per-chunk from a master seed for reproducibility: `chunk_seed = master_seed + chunk_index`

### 3. VAE Decode & Streaming

After each chunk's diffusion completes:

1. VAE-decode the chunk's latents → raw audio waveform (48kHz)
2. **Crossfade** the overlap zone with the previous chunk:
   - Linear crossfade over the 5s overlap (or use a Hann window for smoother blending)
   - This masks boundary artifacts
3. **Emit** the new audio segment via a callback / SSE / WebSocket
4. On the frontend, append to the audio player's buffer

### 4. Lyric Splitting Strategy

Lyrics must be divided per-chunk. Two approaches:

**A. Timestamp-based (preferred if LM provides timestamps):**
- If using CoT/LM mode, lyric timestamps are available
- Split lyrics at the chunk boundary, assigning each line to its chunk

**B. Proportional split (fallback):**
- Estimate total duration per lyric section
- Assign sections to chunks proportionally
- Include ~1 line from the next chunk as overlap context

### 5. Frontend Integration

**UI Changes:**
- New toggle: **"Progressive Generation"** (off by default)
- When enabled, the audio player shows a progress indicator for both current-chunk generation and overall track progress
- Playback begins as soon as chunk 0's audio is ready
- Audio appends seamlessly as subsequent chunks complete

**API Changes:**
- New parameter: `progressive: boolean` in the generation request
- Response mode changes from single-result to **streaming** (SSE or chunked response)
- Each chunk emits: `{ chunk_index, audio_data, is_final, total_chunks }`

### 6. Backend Pipeline

```
progressive_generate(params):
    chunks = plan_chunks(duration, chunk_size=30, overlap=5)
    master_seed = params.seed or random()
    prev_latents = None

    for chunk in chunks:
        # Build per-chunk inputs
        lyrics_slice = split_lyrics(params.lyrics, chunk)
        seed = master_seed + chunk.index

        if chunk.is_first:
            # Standard generation for chunk duration
            outputs = service_generate(duration=chunk.duration, lyrics=lyrics_slice, seed=seed, ...)
        else:
            # Overlap-aware generation
            outputs = service_generate(
                duration=chunk.duration,
                lyrics=lyrics_slice,
                seed=seed,
                # Inject previous chunk's tail latents as frozen context
                target_wavs=prev_latents_tail,
                repainting_start=0.0,
                repainting_end=chunk.overlap_sec / chunk.duration,
                ...
            )

        # VAE decode
        audio = vae_decode(outputs.target_latents)

        # Crossfade with previous chunk
        if prev_audio is not None:
            audio = crossfade(prev_audio_tail, audio, overlap_samples)

        # Stream to client
        emit_chunk(chunk.index, audio, chunk.is_last)

        # Save state for next iteration
        prev_latents = outputs.target_latents
        prev_audio = audio
```

---

## Known Quality Tradeoffs

| Issue | Severity | Mitigation |
|-------|----------|------------|
| No future context within each chunk | Medium | Overlap provides ~5s of bidirectional context from the previous chunk |
| Boundary artifacts | Medium | Crossfade blending over overlap zone |
| Musical structure / phrasing | Medium–High | Lyrics splitting helps; genre/style conditioning stays consistent |
| Endings may be abrupt on non-final chunks | Low | Only the final chunk needs a natural ending; intermediate chunks are continued |
| Inconsistent loudness/dynamics | Low | Post-processing normalization per chunk |

---

## What Needs Investigation

Before implementation, these questions need answers (probably via experimentation):

1. **Optimal overlap duration** — 5s is a guess. Need to test 3s, 5s, 10s and compare boundary quality.
2. **Repainting vs. chunk_masks** — which mechanism produces better overlap blending?
3. **Crossfade window** — linear vs. Hann vs. equal-power crossfade?
4. **Lyric alignment accuracy** — does splitting lyrics mid-chunk cause hallucinated vocals?
5. **Chunk size** — 30s is arbitrary. Does 20s or 45s work better for musical phrasing?
6. **Noise schedule at boundaries** — should the overlap zone use a modified noise schedule for smoother blending?

---

## Implementation Phases

### Phase 1: Backend Proof of Concept
- [ ] Chunk planner utility
- [ ] Per-chunk generation loop (no streaming, just sequential chunks saved to disk)
- [ ] Overlap injection via repainting mechanism
- [ ] Crossfade blending
- [ ] Quality comparison vs. full-sequence generation
- [ ] Determine optimal overlap size

### Phase 2: Streaming Infrastructure
- [ ] SSE or WebSocket endpoint for chunk streaming
- [ ] Per-chunk audio encoding (WAV or Opus segments)
- [ ] Progress reporting per-chunk and overall

### Phase 3: Frontend Integration (React UI)
- [ ] Progressive generation toggle in Create panel
- [ ] Streaming audio player that appends chunks
- [ ] Progress UI showing chunk-level and overall progress
- [ ] Fallback: if progressive fails, retry with full-sequence

### Phase 4: Polish & Optimization
- [ ] Lyric-aware chunk splitting (align to section boundaries)
- [ ] Adaptive chunk sizing based on song structure
- [ ] Memory optimisation (unload previous chunk's intermediates)
- [ ] User preference persistence

---

## Constraints

- **Existing pipeline untouched.** Progressive generation is a completely separate code path, gated behind an opt-in flag. The current `service_generate` → `generate_audio` flow must never be affected.
- **Same model, no retraining.** This approach works with the existing trained DiT weights.
- **GPU memory.** Each chunk is smaller than a full track, so VRAM usage per chunk is actually lower. No additional memory concerns.
