# Future Stem Separation Enhancements

These are planned improvements to the stem separation system. They build on the existing `audio_separator` + multi-track mixer infrastructure.

---

## 1. Silent Stem Filtering

**Goal:** Automatically detect and hide stems that contain no meaningful audio (e.g. a "piano" stem on a track with no piano).

### How it would work

1. After `sep.separate()` returns, analyze each stem's RMS energy using numpy
2. If a stem's average RMS is below a configurable threshold (e.g. -60 dBFS), mark it as "silent"
3. Silent stems are excluded from the results sent to the frontend (so they don't appear in the mixer)
4. The files remain on disk in case a user explicitly wants to download them

### Implementation notes

- ~15 lines of numpy code in `stem_service.py` after each `sep.separate()` call
- Threshold should be configurable (could expose in the UI later)
- Consider analyzing in short windows rather than whole-file RMS to catch stems with brief transients

---

## 2. Stem Reprocessing via ACE-Step Cover/Repaint

**Goal:** Feed an extracted stem back into ACE-Step's cover/repaint system to regenerate it at higher quality. The separated stems are often noisy or have artifacts — ACE-Step could re-synthesize a cleaner version while preserving the musical content.

### Concept

1. Take an extracted stem (e.g. bass, guitar, vocals)
2. Feed it as the source audio to ACE-Step's cover/repaint task
3. Use a text prompt describing the isolated instrument (e.g. "solo bass guitar, clean tone")
4. Use moderate-to-high repaint strength (70-90%) to preserve timing/notes while improving quality
5. Output replaces the original stem in the mixer

### Viability assessment

**Promising:**
- The cover/repaint infrastructure already exists and works well
- High repaint strength preserves musical structure while re-synthesizing timbre
- Could genuinely improve quality of separated stems

**Risks to investigate:**
- ACE-Step is trained on full songs, not isolated instruments — a solo bass track is out-of-distribution and the model may hallucinate other instruments
- Timing drift: even a few ms of drift would make the stem unusable when mixed back with others
- Genre/style mismatch if prompt engineering isn't precise

**Mitigations:**
- Use repaint strength ≥ 80% to keep tight alignment
- Precise instrument-specific prompts ("solo bass guitar, no other instruments")
- Post-process: trim/align regenerated stem to match original's exact length and sample rate
- A/B comparison in the mixer before committing to the regenerated version

### UI concept

- Add a "Reprocess" button per stem in the mixer
- Opens a dialog with repaint strength slider and instrument type hint
- Progress indicator while reprocessing
- Side-by-side toggle (original vs reprocessed) before replacing
