---
description: Research task — investigate how to implement Structure & Variation controls (bars per section + melodic variation)
---

# Investigate: Structure & Variation Controls

> Spotted in another fork of sdbds's ACE-Step project (code not shared).
> Goal: understand feasibility and implement our own version.

## Feature Description

Two controls under a "✦ STRUCTURE & VARIATION" section (marked Experimental):

### Bars Per Section
- **Options:** 4, 8, 16, 32
- **Description:** "How many bars each section (Verse, Chorus, etc.) lasts. 8 = standard, 4 = short, 16/32 = extended."

### Melodic Variation
- **Slider:** 0% → 100% (default shown at 45%)
- **Description:** "Adds randomness to melody generation. 0% = model default, 50%+ = experimental variation. Respects tempo."

## Hypotheses

### Bars Per Section — likely approaches (investigate in order)

1. **Caption/prompt engineering** (highest probability)
   - Prepend structural cues to the style prompt: `"4-bar phrases, short sections"`, etc.
   - The DiT was trained on structurally-annotated music, so it may respond to these cues
   - Easiest to test — just try it manually and listen

2. **BPM × duration × time_signature calculation**
   - Calculate latent frames per bar: `(60 / BPM) × beats_per_bar × model_frame_rate`
   - Force section boundaries at multiples of that frame count
   - Would need to understand the latent frame rate (likely ~50Hz at 44.1kHz)

3. **Latent-level structural constraints**
   - Apply attention masking or latent resets at bar boundaries
   - Would require modifying the DiT inference loop
   - Most invasive, least likely for an "Experimental" tag

### Melodic Variation — likely approaches

1. **Scaled noise injection into latent space**
   - Add Gaussian noise to melodic-frequency latent channels during/after diffusion
   - Selectively perturb pitch dimensions while preserving rhythm/tempo
   - We already have `latent_shift` and `latent_rescale` — this would be similar

2. **DiT sampling temperature**
   - Secondary temperature on the diffusion denoising (not the LM)
   - Makes sampling more stochastic without affecting structural conditioning
   - "Respects tempo" suggests rhythm features are preserved

3. **Guidance scale perturbation**
   - Vary `guidance_scale` per-step or per-channel to introduce controlled randomness
   - Lower guidance = more model creativity = more melodic variation

## Research Steps

- [ ] **Step 1:** Examine `acestep/pipeline_ace_step.py` — find where latent denoising happens
- [ ] **Step 2:** Map latent dimensions — understand which channels encode melody vs rhythm vs timbre
- [ ] **Step 3:** Test caption-based bars control — manually add structural prompts and compare outputs
- [ ] **Step 4:** Test noise injection — add scaled noise to latents and evaluate effect on melody
- [ ] **Step 5:** Test DiT temperature — if the scheduler supports it, try varying diffusion stochasticity
- [ ] **Step 6:** Prototype UI controls and wire them through to generate.ts → api_server.py → pipeline

## Files to Study

| File | Why |
|------|-----|
| `acestep/pipeline_ace_step.py` | Main inference pipeline — where latent denoising happens |
| `acestep/music_dcae/` | VAE/DCAE — maps between audio and latent space |
| `acestep/scheduler_shift.py` | Diffusion scheduler — controls noise schedule |
| `acestep/dit/` | DiT model — transformer that does the actual denoising |
| `acestep/api_server.py` | API — where new parameters would be received |

## Reference

Screenshot of the feature from the other fork is saved in the conversation
(conversation ID: 565c5996-6b67-4a42-81da-b6a22efbfae2).
