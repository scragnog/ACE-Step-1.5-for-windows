# Remaining Wrangler-Inspired Features

Features identified from the [Ace-Step-Wrangler](../Ace-Step-Wrangler) comparison that haven't been implemented yet. See `wrangler_vs_our_frontend.md` in the conversation artifacts for the full analysis.

## High Priority

### Auto Duration (LM-Assisted Estimation)
- **What:** "Auto" button next to the Duration slider that estimates optimal duration from lyrics + BPM
- **Backend:** New `/estimate-duration` endpoint (port logic from Wrangler's `backend/main.py`)
- **Frontend:** Auto button in `MusicParametersSection.tsx`, debounced re-estimation on lyrics/BPM changes
- **Effort:** ~1 day

### Waveform Repaint Region Selector
- **What:** Canvas-based drag-to-select on the waveform for repaint start/end instead of manual number entry
- **Backend:** New `/estimate-sections` endpoint to provide section labels (verse, chorus, etc.)
- **Frontend:** Extend `WaveformVisualizer.tsx` with drag-select, wire to `CoverRepaintSettings.tsx`
- **Effort:** ~2 days

### "Send to Rework" One-Click Flow
- **What:** Button on result cards to load generated audio into Cover/Repaint mode in one click
- **Frontend:** Add "Use as source" button to `SongList.tsx` / `RightSidebar.tsx`, auto-switch task type, populate source audio and waveform
- **Effort:** ~1 day

## Medium Priority

### Friendly Parameter Labels
- **What:** Show "Creativity" and "Quality" labels alongside the raw CFG/steps sliders
- **Frontend:** Label aliases in `GenerationSettingsAccordion.tsx`
- **Effort:** ~2 hours

### VRAM-Aware Batch Constraint
- **What:** Auto-limit batch size based on detected GPU VRAM
- **Backend:** Expose VRAM info via `/system-info` endpoint
- **Frontend:** Cap batch slider max based on response
- **Effort:** ~half day

### Output State Machine
- **What:** Cleaner UI states for idle → generating → results with dedicated panels for each state
- **Frontend:** Refactor result display in `SongList.tsx` to have explicit idle/generating/results states
- **Effort:** ~1 day
