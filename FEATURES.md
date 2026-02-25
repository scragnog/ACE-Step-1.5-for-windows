# New Features

This document tracks all new features added on top of the upstream [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) (qinglong branch).

> **Workflow:** Each feature is developed on a dedicated branch (`feature/<name>`), committed, pushed to [scragnog/ACE-Step-1.5-for-windows](https://github.com/scragnog/ACE-Step-1.5-for-windows), and merged into the default branch (`qinglong`).

---

## Tempo Scale & Pitch Shift (Cover Mode)

**Branch:** `qinglong`  
**Status:** ✅ Merged

Pre-process source audio before VAE encoding with pitch-preserving tempo changes and speed-preserving pitch shifts. Enables changing the tempo of a cover independently from its key, or transposing a male vocal track into a female range (or vice versa) before generation.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/handler/generate_music_request.py` | Time-stretch via `torchaudio.functional.speed()` and pitch shift via `torchaudio.functional.pitch_shift()`, applied after `process_src_audio()` and before padding/VAE encoding |
| `acestep/core/generation/handler/generate_music.py` | `tempo_scale` and `pitch_shift` params threaded through `generate_music()` |
| `acestep/inference.py` | Added to `GenerationParams` dataclass |
| `acestep/api_server.py` | Alias mapping, request model, REST and Gradio handlers |
| `ace-step-ui/server/src/routes/generate.ts` | `GenerateBody` type + request body (gated to cover/repaint/a2a tasks) |
| `ace-step-ui/components/CreatePanel.tsx` | State variables, prop passing, generate request inclusion |
| `ace-step-ui/components/sections/CoverRepaintSettings.tsx` | Side-by-side Tempo Scale (0.5x–2.0x) and Pitch Shift (-12 to +12 semitones) sliders |
| `ace-step-ui/i18n/translations.ts` | Labels, help text, and tooltips for both controls |

### How it works

1. Both sliders appear in the **Cover Settings** section (hidden for text2music/extract modes)
2. **Tempo Scale** (0.5x–2.0x): Uses phase vocoder to change speed without affecting pitch. 1.3x = 30% faster output, 0.8x = 20% slower
3. **Pitch Shift** (-12 to +12 semitones): Transposes the source audio without changing speed. +4 shifts up ~a major third (e.g. male→female vocal range), -3 shifts down a minor third
4. Both transforms are applied to the source audio tensor *before* it enters the padding and VAE encoding pipeline, so the model generates in the new tempo/key space
5. The two can be combined — e.g. speed up 1.2x AND shift up 3 semitones simultaneously
6. Display format: Tempo shows as `1.30x`, Pitch shows as `+3 ♯` / `-2 ♭`

---

## Activation Steering (TADA)

**Branch:** `feature/activation-steering`  
**Status:** ⚠️ Experimental (In-Progress)  

Total integration of Task Adaptive Directional Activation (TADA), enabling zero-shot generation guidance by modifying model activations directly.  
> **Note:** This feature is currently in-progress, experimental, and may not yet work as intended.

### What's included

| File | Description |
|------|-------------|
| `acestep/compute_steering.py` | Core mathematical script to isolate mathematical delta vectors using contrastive decoding. |
| `acestep/steering_controller.py` | Handles PyTorch hook injection and logic for applying multiple chained concepts during auto-regressive generation. |
| `acestep/core/generation/handler/steering_mixin.py` | **[NEW]** Provides high-level handler methods for vector I/O, enabling, and UI configuration mapping. |
| `ace-step-ui/components/sections/ActivationSteeringSection.tsx` | **[NEW]** UI component for computing contrastive vectors, dynamically overriding base prompts, and applying multi-concept guidance. |
| `docs/en/Activation_Steering_Tutorial.md` | **[NEW]** Comprehensive guide on use-cases and terminology. |

### How it works

1. It isolates the explicit mathematical "essence" of a concept by decoding a neutral base prompt and then computing the targeted difference against an activated prompt.
2. The user submits lines of concept modifiers into the **Compute Queue** to generate and cache these arrays on disk as `.pkl` files.
3. Computed vectors can be hot-loaded directly into the model's memory map across targeted layers (`tf6`, `tf7`).
4. Scale (alpha) sliders fine-tune the absolute intensity of each loaded concept dynamically, including supporting negative steering constraints. 
5. Users can selectively unload or permanently **Delete** poor concepts through the UI via the Express backend.

---

## Advanced Guidance & Solver Modes

**Branch:** `feature/pag-dpmsde-tooltips`  
**Status:** ✅ Merged

Total overhaul of the inference backend to support 7 distinct guidance modes and 4 ODE solver algorithms, complete with UI integrations and educational tooltips.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/guidance.py` | **[NEW]** Central registry for Guidance modes (Plain CFG, CFG++, Dynamic CFG, Rescaled CFG, APG, ADG, PAG) |
| `acestep/core/generation/solvers.py` | **[NEW]** Central registry for ODE Step Solvers (Euler, Heun, DPM++ 2M, RK4) |
| `patch_checkpoints.py` | Patches base models on-the-fly to hook into the new guidance/solver registries without touching upstream code |
| `ace-step-ui/components/CreatePanel.tsx` | Added Guidance dropdown, Inference Method dropdown, and conditional PAG detail sliders |
| `ace-step-ui/i18n/translations.ts` | 40+ localized educational tooltips explaining every generation parameter |

### How it works

1. **Guidance Modes:** Choose between strict mathematical CFG variants (Plain, CFG++, Dynamic, Rescaled) or specialized audio-flow projections (APG, ADG) to control how strongly the text guides the music. PAG (Perturbed Attention Guidance) adds structural clarity independently.
2. **Solvers:** Trade off speed vs. quality. Euler (1 eval/step) is fast. Heun (2 evals) and RK4 (4 evals) offer higher-quality numerical integration at the cost of generation speed. DPM++ 2M offers 2nd-order quality at 1 eval per step.
3. Hovering over any parameter reveals a localized tooltip explaining what it does.

---

## One-Click Launcher with Model Selection

**Branch:** `feature/model-loading-enhancements`  
**Status:** ✅ Merged

Single-click launch experience with an interactive loading screen that lets you choose which models to load before the servers start.

### What's included

| File | Description |
|------|-------------|
| `LAUNCH.bat` | One-click launcher — scans `checkpoints/` for available models, writes `loading-config.js`, opens loading screen, starts Express first then Python API |
| `START.bat` | Alternative launcher without loading screen |
| `loading.html` | Animated loading page with model selection dropdowns, 5s auto-continue timer, and real-time service status checklist |
| `ace-step-ui/server/src/routes/models.ts` | `POST /api/models/update-env` — updates `.env` file on disk when the user changes model selection |
| `ace-step-ui/server/src/index.ts` | CORS fix to allow `file://` origin (loading screen runs from local file) |

### How it works

1. `LAUNCH.bat` scans `checkpoints/` for all `acestep-v15-*` (DiT) and `acestep-5Hz-lm-*` (LM) models
2. Writes the model list and current `.env` selections to `loading-config.js`
3. Opens `loading.html` in the browser — dropdowns are pre-populated and pre-selected
4. **Model selection:** If the user changes a dropdown, the loading screen calls `POST /api/models/update-env` (Express) to update `.env` on disk before the Python API reads it
5. **5-second auto-continue timer** — resets if the user interacts with the dropdowns
6. Express starts **before** the Python API (3s head start), allowing `.env` updates to take effect
7. The loading page polls three services:
   - **Python API** — `/v1/models/status` (waits for `active_model` to be non-null)
   - **Express backend** — `localhost:3001/health`
   - **Vite frontend** — `localhost:3000`
8. Auto-redirects to the app once all three are confirmed ready

### API changes

- Added `GET /v1/models/status` endpoint to `acestep/api_server.py` (no auth required, includes `lm_model` field)
- Added `POST /api/models/update-env` endpoint to Express for loading screen `.env` updates
- `ace-step-ui/start.bat` checks `ACESTEP_NO_BROWSER` env var to avoid opening a duplicate browser tab

---

## LM Model Hot-Switching

**Branch:** `feature/model-loading-enhancements`  
**Status:** ✅ Merged

Fixes a bug where changing the Language Model (5Hz LM) in the UI had no effect — the original model loaded at startup was always used regardless of the user's selection. Now supports live switching between LM models during a session.

### What's included

| File | Description |
|------|-------------|
| `acestep/api_server.py` | `_ensure_llm_ready()` rewritten to detect model mismatch and hot-switch (unload old → load new). Added `app.state._llm_model_path` tracking. `/v1/models/status` now returns `lm_model`. |
| `ace-step-ui/server/src/routes/generate.ts` | `lm_model_path` is now always sent to the Python API (previously only sent when `thinking=true`) |
| `ace-step-ui/components/CreatePanel.tsx` | LM model dropdown syncs to the actually loaded model on initial page load via `/api/models/status` |

### How it works

1. Every generation request now includes the selected `lm_model_path` (regardless of thinking mode)
2. `_ensure_llm_ready()` compares the requested model against the currently loaded one (`app.state._llm_model_path`)
3. If they differ, the current LLM is unloaded and re-initialized with the new model (~5–15s blocking operation)
4. If they match, it's a no-op (instant return)
5. On initial UI load, `CreatePanel` fetches `/api/models/status` and syncs the LM dropdown to whichever model is actually loaded

---

## Enhanced Model Selector

**Branch:** `qinglong`  
**Status:** ✅ Merged

Dynamic model discovery and hot-swap switching. The model dropdown auto-populates from all installed checkpoints and supports switching the active DiT model without restarting the server.

### What's included

| File | Description |
|------|-------------|
| `acestep/handler.py` | `switch_dit_model()` — hot-swaps DiT weights, handles LoRA unload, VRAM cleanup, attention fallback |
| `acestep/api_server.py` | `GET /v1/models/list` — scans `checkpoints/` for installed models; `POST /v1/models/switch` — triggers model swap |
| `ace-step-ui/server/src/routes/models.ts` | Express proxy routes for model list, status, and switch |
| `ace-step-ui/server/src/routes/generate.ts` | Updated proxy target to `/v1/models/list` |
| `ace-step-ui/services/api.ts` | `getModels()` / `switchModel()` frontend API methods |
| `ace-step-ui/components/CreatePanel.tsx` | Mismatch banner, switch button, dynamic dropdown |
| `LAUNCH.bat` | Added server TypeScript rebuild step before startup |

### How it works

1. On startup, the Python API scans `checkpoints/` for all `acestep-v15-*` directories
2. The frontend fetches the installed model list via `/api/generate/models`
3. If the selected model differs from the loaded model, an amber mismatch banner appears
4. Clicking "Switch" calls `POST /v1/models/switch` which hot-swaps the DiT model (unloads LoRA, frees VRAM, loads new weights)
5. A fallback list of the 6 default ACE-Step models is shown while the Python API starts

---

## Simple Shutdown

**Branch:** `feature/Simple-Shutdown`  
**Status:** ✅ Merged

Quit button in the sidebar that gracefully shuts down all ACE-Step processes (Python API, Vite frontend, Express backend, and their hosting terminal windows).

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/server/src/index.ts` | `POST /api/shutdown` — snapshots the process table via `Get-CimInstance`, walks the process tree to find ancestor shells, kills everything |
| `ace-step-ui/components/Sidebar.tsx` | Power icon quit button (red styling, appears at sidebar bottom) |
| `ace-step-ui/App.tsx` | ConfirmDialog wiring + "ACE-Step has shut down" overlay |

### How it works

1. Click the red **Quit** button at the bottom of the sidebar
2. A confirmation dialog appears — "Are you sure you wish to shut down ACE-Step?"
3. On confirm, `POST /api/shutdown` is called:
   - Snapshots the entire Windows process table in one `Get-CimInstance Win32_Process` call (~200ms)
   - Finds PIDs on ports 8001 (Python API) and 3000 (Vite) via `netstat`
   - Walks UP the process tree to find ancestor CMD/PowerShell/conhost windows
   - Kills all collected PIDs with `taskkill /F /T`
   - Express process exits last
4. Browser shows a "You may now close this tab" overlay

---

## Persistent Settings

**Branch:** `feature/Persistent-Settings`  
**Status:** ✅ Merged

Toggleable localStorage persistence for all generation settings. Disabled by default — once enabled in Settings, every parameter survives page refresh.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/hooks/usePersistedState.ts` | `usePersistedState` hook — drop-in `useState` replacement with auto-persistence gated by `ace-persist-enabled` flag |
| `ace-step-ui/components/SettingsModal.tsx` | "Persistent Settings" toggle section with enable/disable switch |
| `ace-step-ui/components/CreatePanel.tsx` | ~35 useState calls converted to usePersistedState |

### How it works

1. Open **Settings** → toggle **"Remember my settings"** ON
2. All generation parameters (style, lyrics, BPM, model, LoRA path, inference settings, etc.) are now auto-saved to localStorage
3. Page refresh → all settings restored
4. Toggle OFF → all saved settings cleared, page reloads with defaults
5. Future features: use `usePersistedState('key', default)` instead of `useState(default)` — one-line change

---

## Track List Updates

**Branch:** `feature/Track-List-Updates`  
**Status:** ✅ Merged

A collection of track list UX improvements, bug fixes, and a new bulk-delete feature.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/WaveformVisualizer.tsx` | Replaced upstream fixed-width bar rendering with dynamic spacing so the waveform fills the full progress bar width. Includes shared `AudioContext`, LRU cache (30 entries), and `AbortController` for proper cleanup. |
| `ace-step-ui/server/src/routes/generate.ts` | Fixed generation progress display — parses tqdm-style `progress_text` from the Python API. Added per-job queue detection so queued jobs don't leak the running job's progress. |
| `ace-step-ui/components/CreatePanel.tsx` | "Queue Next" button now uses i18n key instead of hardcoded English text. |
| `ace-step-ui/i18n/translations.ts` | Added `queueNext`, `deleteAllTracks`, `deleteAllTracksConfirm`, `allTracksDeleted`, `deleteAllFailed` keys for all 4 languages (en, zh, ja, ko). |
| `ace-step-ui/components/SongList.tsx` | Removed `createdAt` DESC sort from `listItems` so songs maintain their order from state. Added `onDeleteAll` prop with a trash icon button in the header bar. |
| `ace-step-ui/App.tsx` | Removed the `refreshSongsList` sort that caused completed songs to jump to the top. Completion now does in-place merge instead of full list reload. Added `handleDeleteAll` with confirmation dialog. |
| `ace-step-ui/server/src/routes/songs.ts` | Added `DELETE /api/songs/all` endpoint — deletes all user songs and associated audio/cover files from storage. |
| `ace-step-ui/services/api.ts` | Added `deleteAllSongs()` API client method. |
| `LAUNCH.bat`, `START.bat`, `ace-step-ui/start.bat`, `ace-step-ui/start-all.bat` | Added `/min` flag to all `start` commands so spawned terminal windows launch minimized. |

### Changes in detail

- **Waveform alignment** — Bars now fill the entire progress bar width using dynamic step calculation instead of fixed `barWidth=2, gap=1`.
- **Generation progress** — The Express backend now parses tqdm output (`14%|##5| 27/200 [00:06<00:42, 4.10steps/s]`) to extract percentage, ETA, and step count.
- **Queue progress bleed fix** — The Python API maps both `queued` and `running` to status code `0` and shares a global `log_buffer.last_message`. The Express status handler now checks the per-job `stage` field in the result data to detect queued jobs and returns `status: 'queued'` with `progress: 0` instead of leaking the running job's tqdm output.
- **Track reordering fix** — Two sorts were causing completed songs to jump to the top: one in `refreshSongsList` (App.tsx) and one in `listItems` (SongList.tsx). Both removed. Completion now does an in-place merge that preserves existing order.
- **Delete All Tracks** — Trash icon button in the track list header (next to Select/Filter). Shows a confirmation dialog with the count of tracks. Calls `DELETE /api/songs/all` which deletes all audio/cover files from storage and removes all DB rows.
- **Minimized windows** — All spawned terminal windows (Python API, Express, Vite) now launch minimized via `/min` flag.

---

## Advanced Multi-Adapter System

**Branch:** `feature/advanced-adapters`  
**Status:** 🚧 In Progress

Slot-based multi-adapter loading (up to 4 simultaneous LoRA/LoKr adapters) with per-slot scaling and per-module-group scaling (Self-Attn, Cross-Attn, MLP). Uses weight-space merging approach. Existing basic single-adapter UI is preserved — the advanced system is behind an opt-in "Advanced" checkbox.

### What's included

| File | Description |
|------|-------------|
| `acestep/core/generation/handler/lora/advanced_adapter_mixin.py` | **[NEW]** `AdvancedAdapterMixin` — delta extraction, weight-space merging (`base + Σ(scale × group_scale × delta)`), slot management |
| `acestep/core/generation/handler/lora_manager.py` | Import + export `AdvancedAdapterMixin` |
| `acestep/handler.py` | Added `AdvancedAdapterMixin` to MRO, init state (`_adapter_slots`, `_next_slot_id`, `_merged_dirty`, `lora_group_scales`) |
| `acestep/api_server.py` | 3 new endpoints, updated `load`/`unload` for slot param, new request models |
| `ace-step-ui/server/src/routes/lora.ts` | 3 new routes: `GET /list-files` (folder scanner), `POST /group-scales`, `POST /slot-group-scales` |
| `ace-step-ui/services/api.ts` | `listLoraFiles()`, `setGroupScales()`, `setSlotGroupScales()`, updated `loadLora`/`unloadLora` for slot support |
| `ace-step-ui/components/CreatePanel.tsx` | Advanced toggle, folder browser, slot cards with per-slot scale + expandable per-group sliders |

### How it works

1. Open the **LoRA** panel and check **Advanced (Multi-Adapter)**
2. Enter an adapter folder path → click **Scan** → available `.safetensors` files appear
3. Click **Load** on any adapter → it's loaded into a slot, delta extracted via weight-space merging
4. Each slot card shows: adapter name, type badge (LoRA/LoKr), overall scale slider (0–2)
5. Expand **Groups** on a slot → independent Self-Attn, Cross-Attn, MLP sliders (0–2)
6. Load additional adapters (up to 4) — all merge simultaneously: `decoder = base + Σ(slot_scale × group_scale × delta)`
7. Per-adapter group scale settings are persisted in localStorage by adapter filename
8. Uncheck "Advanced" → original basic single-adapter UI appears unchanged

### Architecture note

Basic mode uses PEFT runtime hooks (existing). Advanced mode uses **weight-space merging**: backs up base decoder to CPU (~1.5GB), extracts each adapter as a delta, applies `base + Σ(scaled deltas)` at inference. Re-merge takes ~1s on scale change.

---

## Creation Panel Reorganization

**Branch:** `feature/cot-accordion`  
**Status:** ✅ Merged

Total UX reorganization and architectural refactoring of the Create panel to reduce cognitive overload and group related settings.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/CreatePanel.tsx` | Massively refactored into a layout shell delegating to ~13 modular sub-components |
| `ace-step-ui/components/accordions/*` | **[NEW]** Generation Settings, Track Details, Adapters, Score System, Expert Controls, Guidance Settings, LmCot Accordions |
| `ace-step-ui/components/sections/*` | **[NEW]** Audio Selection, Lyrics, Style, Music Parameters, Cover Repaint Settings, Task Type, Simple Mode Settings, Audio Library Modal |

### How it works

1. **Categorized Settings:** Options are now grouped into logical accordions (Generation Settings, Expert Controls, Adapters, Score System) rather than a single massive scrolling list.
2. **Track Details Isolation:** Lyrics, Style, and Music Parameters are cleanly nested under a Track Details accordion in Custom Mode.
3. **Simple vs Custom Mode:** Simple mode presents a cleaner top-level interface while Custom mode exposes deep configuration.
4. **Tooltips & i18n:** Every single parameter now features a localized tooltip explaining its function.
5. **Maintainability:** The massive 4500-line CreatePanel was decomposed into specific, maintainable UI sections and components.

---

## JSON Export & Import

**Branch:** `feature/json-export-import`  
**Status:** ✅ Merged

Export all generation parameters to a JSON file and import them later to reproduce exact configurations. Includes full adapter and steering parameter persistence in the Generation Parameters sidebar.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/CreatePanel.tsx` | Export/Import buttons below Task Type selector. `handleExportJson` serializes all state to a downloadable `.json` file. `handleImportJson` parses the file and restores all state variables, auto-opening steering/adapter panels if applicable. |
| `ace-step-ui/App.tsx` | Replaced 65-line cherry-picked parameter allowlist in `handleGenerate` with `{ ...params }` spread — ensures all current and future parameters flow through to the API automatically. |
| `ace-step-ui/server/src/routes/generate.ts` | Simplified to `const params = req.body as GenerateBody` — stores the entire frontend payload verbatim in the SQLite `params` column. |
| `ace-step-ui/components/RightSidebar.tsx` | Added adapter slot display (name, type, scale) and steering concept display (concept, alpha) to the Generation Parameters sidebar. Removed redundant "Use ADG" entry (now covered by Guidance Mode). |
| `ace-step-ui/services/api.ts` | Synced `GenerationParams` interface with adapter and steering fields, updated `inferMethod` type union. |
| `ace-step-ui/types.ts` | Added `loraLoaded`, adapter, and steering properties to the shared `GenerationParams` interface. |

### How it works

1. **Export:** Click the **Export JSON** button → all generation parameters (including adapter slots, steering concepts, guidance mode, PAG settings, etc.) are serialized and downloaded as a timestamped `.json` file.
2. **Import:** Click the **Import JSON** button → select a previously exported file → all parameters are restored. If the config included loaded adapters or steering concepts, those panels auto-expand.
3. **Sidebar Display:** After generation, the right sidebar now shows:
   - **Basic LoRA** name and scale (only when not using advanced adapters)
   - **Advanced Adapter Slots** with name, type badge, and scale per slot
   - **Steering Concepts** with concept name and alpha value
4. **Future-proof API:** `App.tsx` now spreads params directly, so any new fields added to `CreatePanel` automatically flow through without needing to update the allowlist.

---

## Debug Panel & UI Polish

**Branch:** `feature/debug-panel`  
**Status:** ✅ Merged

Live system monitoring panel and UI polish improvements: a collapsible debug panel showing GPU VRAM, RAM, and CPU usage alongside a real-time streaming API log, a resizable Create Panel, and a streamlined sidebar toggle.

### What's included

| File | Description |
|------|-------------|
| `acestep/api_server.py` | Extended `LogBuffer` with ring buffer + cursor. Added `GET /v1/system/metrics` (GPU, RAM, CPU) and `GET /v1/system/logs` (cursor-based). |
| `ace-step-ui/server/src/routes/system.ts` | **[NEW]** Express route: metrics proxy + SSE log stream at `/api/system/*`. |
| `ace-step-ui/server/src/index.ts` | Mounted system routes. |
| `ace-step-ui/components/DebugPanel.tsx` | **[NEW]** Fixed right-edge panel with progress bars, retro terminal log viewer, persisted state. |
| `ace-step-ui/App.tsx` | Integrated DebugPanel, content shift on open, resizable CreatePanel with drag handle. |
| `ace-step-ui/components/Sidebar.tsx` | Replaced logo + separate arrow with a single purple circle toggle (chevron arrow). |
| `requirements.txt` | Added `psutil>=5.9.0` for CPU/RAM metrics. |

### How it works

1. **Debug Panel:** A toggle tab on the right screen edge opens a 400px panel showing VRAM/RAM/CPU metrics (polled every 2s) and a streaming API log with color-coded levels (green text on black — retro terminal style). Panel state persists across sessions.
2. **Content Shift:** When the debug panel opens, the entire layout (including Song Details sidebar) smoothly slides left to keep everything visible.
3. **Resizable Create Panel:** Drag the right edge of the parameters panel to resize it (280–600px). Width persists across sessions. The track list absorbs the change.
4. **Sidebar Toggle:** The purple circle in the top-left now contains a chevron arrow that rotates to indicate expand/collapse. The separate arrow button has been removed.

---

## Stem Extraction (Extract Mode)

**Branch:** `feature/extract-task`  
**Status:** ✅ Merged

Full stem extraction workflow using ACE-Step's generative extract task. Select one or more instrument stems to isolate from a source audio file — each creates a separate queued job. Includes quality presets, style hints, and lyrics guidance for vocal tracks.

### What's included

| File | Description |
|------|-------------|
| `ace-step-ui/components/sections/ExtractTrackSelector.tsx` | Multi-select toggle chip UI for choosing stems (12 track types) |
| `ace-step-ui/components/CreatePanel.tsx` | Extract mode flow: quality presets, style hint, lyrics guidance, meta clearing, handleGenerate loop for multi-track jobs |
| `ace-step-ui/server/src/routes/generate.ts` | Passes `src_audio_path`, `reference_audio_path`, `track_name` to Python backend |
| `ace-step-ui/server/src/routes/referenceTrack.ts` | Extended upload whitelist for `.ogg`, `.opus`, `.webm` formats |
| `acestep/api_server.py` | Whitelisted project audio directory in `_validate_audio_path` |
| `acestep/core/generation/handler/io_audio.py` | Replaced `torchaudio.load` with `soundfile.read` for Windows compatibility |
| `ace-step-ui/i18n/translations.ts` | 20+ localized keys for extract UI (en, zh, ja, ko) |

### How it works

1. **Track Selection:** Toggle one or more stems from 12 available track types (Vocals, Backing Vocals, Drums, Bass, Guitar, Keyboard, Strings, Synth, Brass, Woodwinds, Percussion, FX). Each selected track queues a separate extraction job.
2. **Quality Presets:** Three one-click presets configure inference steps, solver, and guidance mode:
   - ⚡ **Low (Quick):** 20 steps, Euler, Dynamic CFG
   - ⚖️ **Medium:** 50 steps, Heun, Dynamic CFG
   - 💎 **High (Slow):** 200 steps, RK4, Dynamic CFG
3. **Style Hint (Optional):** A text field to describe the expected timbre/genre (e.g., "distorted electric guitar, heavy rock") — passed as the `style` parameter to guide generation quality.
4. **Lyrics Guidance (Optional):** For vocal/backing vocal tracks, paste lyrics to improve extraction accuracy. The `instrumental` flag is automatically set to `false` for vocal tracks.
5. **Meta Clearing:** BPM, key, and time signature are zeroed for extract mode so stale values from previous text2music sessions don't interfere — the model relies on the actual source audio.
6. **Title Format:** Extract jobs are titled `"Vocals - My Song.mp3"` instead of generic names, using the source audio filename.
7. **Windows Fix:** Replaced `torchaudio.load` with `soundfile.read` in the audio processing pipeline, resolving `torchcodec` dependency failures on Windows.

> **Note:** ACE-Step's extract is *generative*, not subtractive. Unlike traditional source separation tools (Demucs, BSRNN), the model re-generates what it thinks each stem sounds like based on the source audio and instruction. This means vocal tracks may occasionally hallucinate audio in silent sections.

---

<!-- 
## [Next Feature Name]

**Branch:** `feature/...`  
**Status:** 🚧 In Progress / ✅ Merged

Brief description.

### What's included
- ...

### How it works
- ...
-->
