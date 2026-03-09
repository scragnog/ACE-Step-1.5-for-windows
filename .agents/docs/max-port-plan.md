# MAX Fork Port Plan

> **Source:** [ElWalki/Ace-Step-MAX (master)](https://github.com/ElWalki/Ace-Step-MAX/tree/master)  
> **Compared on:** 2026-03-01  
> **Full comparison:** See `ace_step_max_comparison.md` in the brain artifacts dir

## Attribution

> **⚠️ IMPORTANT:** All features ported from this list originate with **ElWalki** and his **Ace-Step-MAX** project.
> When any of the following features are documented in `README.md` or `FEATURES.md`, they must include attribution:
>
> ```
> Based on work by [ElWalki](https://github.com/ElWalki/Ace-Step-MAX/tree/master)
> ```
>
> This applies to: feature descriptions, tooltips, changelog entries, and any public-facing documentation about the ported feature.

---

## 🎯 Start Here — Current Priorities

User-selected features to implement first, in order:

| # | Feature | Type | Effort |
|---|---------|------|--------|
| 1 | **MelodicVariation → lmRepetitionPenalty** | Port from MAX | Low |
| 2 | **PMI Quality Scoring / Test-Time Scaling** | Port from MAX | Medium |
| 3 | **Interactive Section Builder (Udio-style)** | Original (inspired by MAX) | High |

> For the Section Builder: the first sub-task is porting MAX's `section-planner.ts` (pure TypeScript, no backend), then building the interactive UI on top of our existing `repaint` task.

---

## Port Candidates

The following features have been verified as present in MAX's codebase and absent from ours.
Priority is unset — to be determined by the user.

---

### UI / UX Fixes (Low Effort)

- [ ] **Cover Mode Bugfix** — Clearing source audio must reset `taskType` to `text2music`. Safety guards in both Simple and Expert generate calls  
  *Files in MAX:* `components/CreatePanel.tsx`

- [ ] **Time Signature Labels** — Show `2/4` / `3/4` / `4/4` / `6/8` instead of raw integers in the dropdown  
  *Files in MAX:* `components/CreatePanel.tsx`

- [ ] **Key Scale Dropdown Fix** — Correctly captures note + mode (e.g. `G major`, `C# minor`) from `onChange` events in both Simple and Expert mode  
  *Files in MAX:* `components/CreatePanel.tsx`

- [ ] **LoRA Quick Unload Button** — A visible Unload button in the LoRA section header even when the LoRA panel is collapsed. Green pulsing dot to indicate loaded state  
  *Files in MAX:* `components/CreatePanel.tsx` ~line 2800

---

### Generation Quality

- [ ] **MelodicVariation → lmRepetitionPenalty** — A slider (0–100%) that scales `lm_repetition_penalty` from 1.0 → 1.5, applied to the 5Hz LM during audio code token generation to discourage repeated melodic patterns. Exposed as a frontend control and passed through to the Python API  
  *Files in MAX:* `components/CreatePanel.tsx` line 2206, `llm_inference.py`

- [ ] **PMI Quality Scoring / Test-Time Scaling** — After generation, scores audio codes against the original prompt using Pointwise Mutual Information via the LLM. Enables best-of-N selection. Weights: Caption 50%, Lyrics 30%, Metadata 20%  
  *Files in MAX:* `ACE-Step-1.5_/acestep/test_time_scaling.py` (411 lines)

---

### LoRA Management

- [ ] **Floating LoRA Manager Panel** — Draggable floating panel (`LoraManager.tsx`) for browsing all LoRAs. Features: search, favorites (localStorage), base model badges, variant/checkpoint selection, right-click menu (Activate / Open Folder), activation dialog with scale slider. Scale capped at 1.0  
  *Files in MAX:* `components/LoraManager.tsx` (new, 31KB), `components/CreatePanel.tsx`, `services/api.ts`, `server/src/routes/lora.ts`

- [ ] **LoRA CPU Backup on Load** — When a LoRA is loaded, back up the full base decoder to CPU (~10GB freed). Our weight-space merging approach only backs up a ~1.5GB subset; this is a larger savings approach for PEFT-mode LoRA  
  *Files in MAX:* `ACE-Step-1.5_/acestep/handler.py`

- [ ] **VRAM Diagnostic API** — `GET /v1/vram/diagnostic` does a deep VRAM scan across all model components and detects LoRA stacking. `POST /v1/vram/cleanup` force-runs `gc.collect()` + `torch.cuda.empty_cache()`  
  *Files in MAX:* `ACE-Step-1.5_/acestep/api_routes.py`

---

### Metadata & Library

- [ ] **ID3 Metadata Auto-Tagging** — Tags every generated MP3 with Title, Artist, BPM, Key (e.g. `G major`), Genre, Encoded-By, and Comment (gen params summary) using `node-id3`. Applied server-side before the file is stored  
  *Files in MAX:* `server/src/services/audioMetadata.ts`, `server/src/routes/generate.ts`

- [ ] **Generation Config Viewer** — Right-click any song → a modal shows all parameters used to generate it (model, steps, seed, BPM, key, etc.)  
  *Files in MAX:* `components/GenerationConfigModal.tsx` (new)

- [ ] **Edit Metadata Modal** — Right-click any owned song → edit Title, Style/Genre, BPM, Key, Time Signature post-generation. Saved to DB immediately, updates song list in real-time. Does not re-tag the audio file  
  *Files in MAX:* `components/EditMetadataModal.tsx` (new), `server/src/routes/songs.ts` (PATCH endpoint)

---

### Extended Presets

- [ ] **Extended Preset System v2** — Presets now capture the full parameter set including: `instruction`, `seed`, `randomSeed`, `melodicVariation`, `lmRepetitionPenalty`, `useCotMetas`, `useCotCaption`, `useCotLanguage`, `useAdg`, `cfgIntervalStart/End`, `taskType`, `repaintingStart/End`, `audioCoverStrength`, `allowLmBatch`, `getScores`, `getLrc`, `scoreScale`, `lmBatchChunkSize`. Backward compatible with old presets  
  *Files in MAX:* `components/CreatePanel.tsx`

---

### Audio Pipeline

- [ ] **Vocal Tab (Demucs as Reference Source)** — A dedicated "Vocal" tab in the audio reference section. Two paths: (1) pick a song from the library and separate vocals via htdemucs_ft, or (2) upload a pre-separated acapella. Separated vocal becomes available as reference audio and/or source for cover mode. Generation is blocked while Demucs runs (VRAM safety)  
  *Files in MAX:* `components/CreatePanel.tsx` (Vocal tab UI), `server/scripts/separate_audio.py`, `server/src/routes/training.ts`

- [ ] **Section Planner (Batch Auto-Pilot)** — Given full lyrics tagged with `[Verse]` / `[Chorus]` / `[Bridge]` etc. and a BPM, calculates a musically-correct duration per section (snapped to 4-measure phrase boundaries) and generates the **entire song automatically**, section by section, stitching audio together. One trigger → one output with no user input between sections.  
  *Files in MAX:* `server/src/services/section-planner.ts` (299 lines)  
  *Note: This is the batch mode. See "Original Features" below for the interactive Udio-style variant we want to build on top of this.*

---

### Backend Stability

- [ ] **Queue Robustness Fixes** — Three targeted fixes to the Express→Gradio pipeline: (1) `processQueue()` wrapped in `try/finally` so the queue flag always resets, (2) failed jobs are marked `'failed'` with error message in the catch block, (3) Gradio client auto-validates connection age (5min) and resets on predict failure  
  *Files in MAX:* `server/src/services/acestep.ts`, `server/src/services/gradio-client.ts`

---

### Advanced / Research

- [ ] **DiT Alignment Score (alternative LRC method)** — Generates LRC timestamps using the DiT's own cross-attention matrices rather than a separate ASR pass. Uses Numba-JIT DTW for alignment. Also includes quality scoring (Coverage, Monotonicity, Confidence). We already have working LRC from the Python API; this is an alternative, potentially higher-quality approach  
  *Files in MAX:* `ACE-Step-1.5_/acestep/dit_alignment_score.py` (870 lines)

- [ ] **Singer Library (Type A — Timbre Embedding)** — System for capturing a singer's vocal timbre as a `.pt` embedding file from an acapella recording using `TimbreEncoder`. Loaded at inference as an additional conditioning input. Full spec exists; Python implementation not yet done in MAX  
  *Files in MAX:* `SINGER_LIBRARY_SPEC.md` (spec), `extract_singer_embedding.py` (extraction script)

- [ ] **Prepare for Training Modal** — Right-click → open a modal to configure and kick off LoRA training data preparation from an existing song. UI exists; backend training pipeline integration is beta  
  *Files in MAX:* `components/PrepareTrainingModal.tsx` (27KB)

---

### Utilities

- [ ] **BPM/Key Detection Script** — Librosa-based offline `detectar_bpm_clave.py` utility for auto-detecting BPM and musical key from an audio file (we have Essentia for this, but this is an alternative)  
  *Files in MAX:* `detectar_bpm_clave.py`

- [ ] **Whisper Lyric Transcription** — Whisper-based scripts (`transcribir_letras*.py`) for auto-transcribing lyrics from audio files  
  *Files in MAX:* `transcribir_letras.py`, `transcribir_letras_v2.py`

---

## Original Features (Inspired By MAX, Built By Us)

Features we want that MAX doesn't have, but that his work informed or partially enables.

- [ ] **Interactive Section Builder (Udio-style)** — The workflow we actually want: generate **one section at a time**, stop and let the user listen, generate **N variations** for the next section using the previous section's audio as source (via ACE-Step's existing `repaint` task), let the user **pick their favourite**, then chain forward. Each accepted section becomes the audio source for the next continuation.

  **How the pieces fit:**
  - **Duration math** → port MAX's `section-planner.ts`. It already calculates per-section durations from BPM + lyric line count + time signature, snapped to phrase boundaries.
  - **Audio continuation** → ACE-Step's `repaint` task already generates audio starting at a given timestamp using an existing file as source. This is the continuation engine.
  - **The gap** → a UI workflow that: exposes sections as steps, fires off N parallel continuation jobs per step, presents the N candidates, and chains the accepted one into the next step.

  **Rough implementation plan:**
  1. Port `section-planner.ts` as-is — pure TypeScript utility, no backend changes
  2. New "Section Builder" UI panel (new component): shows the song as a vertical step-list, each section showing its calculated duration
  3. "Generate" on section N fires N repaint jobs (configurable 1–4 variations), using the stitched audio so far as source, from the cumulative end timestamp
  4. After completion, thumbnails of each variation with a play button and a "Use this" confirm
  5. On confirm, the chosen audio is stitched and the next section becomes active

  *Attribution: Section planner algorithm by [ElWalki](https://github.com/ElWalki/Ace-Step-MAX/tree/master). Interactive workflow is original.*

---

## Explicitly Out of Scope

These exist in MAX but are architectural decisions we're not adopting:

- **User Account System / JWT auth** — MAX is designed as a multi-user web service. We're single-user local-first.
- **Social features / oEmbed sharing** — Tied to their user account architecture.
- **Playlist / Search / User Profile pages** — Dependent on multi-user DB schema.
