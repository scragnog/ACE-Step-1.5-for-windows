# ACE-Step 1.5 for Windows — Enhanced Fork

An enhanced fork of [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) with a rebuilt UI experience, multi-adapter support, and quality-of-life improvements for music generation with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).

<img width="2062" height="952" alt="image" src="https://github.com/user-attachments/assets/6e682194-99f2-4267-b412-1b5198720b87" />

---

## ✨ New Features

> Full details and implementation notes in [FEATURES.md](FEATURES.md).

### 🎵 Melodic Variation
A **Melodic Variation** slider in the Create panel adds controlled melodic randomness to generation by adjusting the LM's repetition penalty (0.0–2.0). Higher values push the language model to explore less-repeated patterns, introducing more melodic variety and structural unpredictability. Lower values reinforce repetition and structural consistency.

### 🏆 Quality Scoring (PMI + DiT Alignment)
Automatic quality scoring system displayed alongside every generated track. Each generation reports two metrics: **PMI Score** (Pointwise Mutual Information — measures lyric-audio semantic coherence, displayed as a percentage) and **DiT Score** (alignment between the DiT and LM outputs, displayed as a 1–5 star rating). Scores appear in the generation info panel and song details sidebar. Toggle on/off via the Score System accordion.

### ⏹ Job Cancellation & Queue Management
Cancel any running or queued generation job without restarting the server. A **Cancel** button appears on each job in the track list during generation. The Python backend exposes a `POST /v1/cancel/{job_id}` endpoint that sends a cooperative stop signal, allowing the current inference step to complete cleanly before halting. Queued jobs are cleared immediately.

### ⬆️ Upscale to HQ
Re-run inference on a previously generated track at higher quality settings without starting from scratch. The **Upscale to HQ** option in a song's dropdown menu re-submits the original generation parameters with increased inference steps, preserving the audio codes from the original run to guide the upscale pass. Configure the HQ step count in Settings. Useful for previewing at low steps (20–50) then upgrading favorites to high-fidelity (100–200 steps).

### 🧠 Advanced Guidance & Solvers
Total control over the generation pipeline with 7 unique mathematical guidance modes (APG, ADG, PAG, Plain CFG, CFG++, Dynamic CFG, Rescaled CFG) and 4 ODE solver algorithms (Euler, Heun, DPM++ 2M, RK4). Includes 40+ multilingual educational tooltips explaining every generation parameter.

### 📐 Timestep Scheduler
Control *where* denoising steps are concentrated across the noise schedule with 6 pluggable schedulers: **Linear** (default, uniform spacing), **DDIM Uniform** (log-SNR uniform, S-shaped distribution), **SGM Uniform** (Karras σ-ramp ρ=7, moderate front-loading), **Bong Tangent** (front-loads structural decisions), **Linear Quadratic** (more budget for fine detail refinement), and **Composite (2-Stage)** — pick different schedulers for the structural (high-noise) and detail (low-noise) phases with configurable crossover point and step split. Composes naturally with the Shift slider and solver selection.

### 🎯 Activation Steering (TADA)
> ⚠️ **Experimental Feature:** Currently in-progress and may not work as intended.

Guide generation without expanding prompt length using isolated internal brain state direction vectors. Includes a UI component to compute steering vectors for multiple concepts at once, override base genres, fine-tune strength (alpha) and model layer targets on-the-fly, and remove vectors directly from disk. See the [Activation Steering Tutorial](docs/en/Activation_Steering_Tutorial.md).

### 🎛️ Advanced Multi-Adapter System
Load up to **4 LoRA/LoKr adapters simultaneously** with independent per-slot scale sliders, per-module-group scaling (Self-Attn, Cross-Attn, MLP), and **per-layer scaling** (layers 0–23). The **Role Blend** panel exposes three human-friendly sliders used simultaneously — 🎤 **Voice** (layers 0–7), 🎸 **Style** (layers 8–15), and 🔗 **Coherence** (layers 16–23) — derived from empirical layer ablation experiments. Uses weight-space merging for zero-hook inference. Per-adapter settings persist across sessions. Includes a built-in **file browser** for scanning and loading `.safetensors` files from a configurable folder.

### 🚀 One-Click Launcher with Model Selection
Double-click `LAUNCH.bat` → an interactive loading screen lets you choose which DiT and LM models to load via dropdowns (auto-populated from your `checkpoints/` folder). Changes are saved to `.env` before the Python API starts. A 5-second auto-continue timer proceeds automatically if you don't interact. All three services (Python API, Express backend, Vite frontend) are monitored and auto-redirect when ready.

### 🔄 Hot-Swap Model Selector
Live model switching without restarting the server. The dropdown auto-discovers all installed checkpoints and shows a mismatch banner if the selected model differs from the loaded one. **LM model switching** is also supported — changing the 5Hz LM model in the Create panel triggers an automatic unload/reload cycle.

### 💾 Persistent Settings
All generation settings (style, lyrics, BPM, model, adapter paths, scales, inference params) survive page refresh via localStorage. Toggle on/off in Settings.

### 🎛️ Creation Panel Reorganization
Total UX overhaul of the Create panel. Settings are now cleanly grouped into collapsible accordions (Generation Settings, Expert Controls, Audio Adapters, Score System), reducing cognitive overload. Simple Mode provides a streamlined interface, while Custom Mode hides complex track formulation variables (Lyrics, Style, Parameters) inside a tidy Track Details section. Every single parameter now features a localized tooltip explaining its function.

### 🎵 Track List Improvements
- Full-width waveform visualizer with shared AudioContext and LRU cache
- Real-time generation progress (parsed from tqdm output)
- Queue system with per-job progress isolation
- Bulk delete all tracks
- Tracks maintain chronological order (no jumping on completion)

### ⏻ Simple Shutdown
Quit button in the sidebar gracefully shuts down all processes (Python API, Vite, Express, and their hosting terminal windows) with a single click.

### 📋 JSON Export & Import
Export all generation parameters to a shareable `.json` file and import them later to reproduce exact configurations. Includes full adapter slot details, steering concepts, and all expert parameters in the Generation Parameters sidebar.

### 🎼 Stem Extraction (Extract Mode)
Isolate individual stems from any audio file using ACE-Step's generative extract task. Select multiple tracks (Vocals, Bass, Guitar, Drums, etc.) and each queues a separate job. Three quality presets (Low/Medium/High) configure optimal solver and step combinations. Optional **Style Hint** guides timbre (e.g., "distorted electric guitar") and **Lyrics Guidance** improves vocal extraction accuracy. Stale metadata is automatically cleared to prevent interference.

### 🎚️ Tempo Scale & Pitch Shift (Cover Mode)
Pre-process source audio before generation with two independent controls: **Tempo Scale** (0.5x–2.0x) changes speed without affecting pitch using phase vocoder, and **Pitch Shift** (-12 to +12 semitones) transposes the key without changing speed. Perfect for making a male vocal track work in a female range (+3–5 semitones) or adjusting cover tempo independently from melody. Both can be combined simultaneously.

### 🎛️ Server-Side Stem Separation
Professional-grade audio stem separation powered by BS-RoFormer (SDR 12.97) and Demucs, with 4 separation modes: Vocals Only, 4-Stem, 6-Stem, and Two-Pass (best quality). Results appear in a **synchronized multi-track mixer** with per-stem volume, mute/solo, and download controls. Models are lazy-downloaded on first use (~1.8 GB). ACE-Step models are automatically offloaded to CPU during separation and restored to GPU after, preventing VRAM exhaustion.

### 🎛️ Auto-Mastering & Mastering Console
Every generated track is automatically run through a professional mastering chain applying multi-band EQ shaping, harmonic saturation, stereo widening, dynamic compression, and loudness maximization via a peak limiter. By default, it uses a profile learned from professional reference audio. 
- **Mastering Console:** Click the sliders icon on any track to open the interactive **Mastering Console**. Tweak EQ bands, drive, width, threshold, and gain in real-time. Each slider includes educational tooltips to help you avoid clipping and distortion.
- **Persistent Settings:** Your custom mastering settings are saved globally and automatically applied to all future generated tracks.
- **Remastering:** Click **Remaster** on any existing track to process the original raw audio with new console settings without re-generating from scratch.
- **Download Options:** When downloading an auto-mastered track, the download modal allows you to select whether you want the **Mastered** version, the raw uncompressed **Original** output from the diffusion model, or **Both**.

### ✨ Audio Enhancement Studio *(Legacy)*
> ⚠️ **Deprecated:** The Auto-Mastering feature above replaces this for most use cases. This tool remains available for users who want manual per-stem DSP control.

Post-processing engine ported from [ComfyUI-Audio_Quality_Enhancer](https://github.com/ShmuelRonen/ComfyUI-Audio_Quality_Enhancer). Apply multi-band EQ (clarity, warmth, air/brilliance, dynamics), reverb (synthetic IR convolution), echo, and stereo widening (mid/side + Haas effect) to any track. **Two modes:** Simple (full-mix DSP via [pedalboard](https://github.com/spotify/pedalboard)/scipy) and Stem Separation (Demucs splits → per-stem targeted enhancement → remix). Comes with **6 built-in presets** (Radio Ready, Warm & Rich, Bright & Clear, Club Master, Lo-Fi Chill, Cinematic) and full manual control. Accessible from any song's dropdown menu → "Enhance Audio".

### 🔀 A/B Track Comparison
Side-by-side comparison of any two generated tracks. Right-click to assign Track A and Track B, then click **Play Comparison** to start synchronized dual-audio playback. Toggle between tracks instantly while maintaining position — the inactive track plays muted in the background. Fully pause-aware: pausing the player pauses both audio elements, and toggling A/B while paused stays paused. Includes a **Diff** button to compare all generation parameters side-by-side.

### 🎤 Synced Lyrics & Song Structure
Real-time **LRC lyrics overlay** on the art box visualizer synced to playback. A collapsible **Lyrics Bar** at the bottom of the song list shows one lyric line at a time with smooth fade-up transitions — expanded by default, hidden during A/B comparison. **Section markers** (Verse, Chorus, Bridge, Outro, etc.) from the LRC file are displayed in a thin row above the player waveform, positioned at their timestamps.

### 🎨 Visualizer Preset Selection
Choose which visualizer presets are included in random rotation via a checkbox grid in **Settings → Visualizer**. Multiple visualizer instances (art box, song list background, fullscreen) coordinate to never show the same preset simultaneously. Default pool: NCS Circle, Spectrum, Mirror, Analog.

### 🔬 Layer Ablation Lab *(Developer Mode)*
Systematically explore what each adapter layer contributes to the generated audio. An automated **Ablation Sweep** generates one track per transformer layer (layers 0–23) with that layer zeroed, allowing you to chart RMS energy delta vs. layer index and identify voice, style, and coherence roles. Manual per-layer sliders, bulk zero/reset controls, and an **Audio Diff** tool (RMS energy comparison between two tracks) are also included.

### 📂 Native Folder Picker
The **Browse** buttons on both basic and advanced adapter panels now open a native Windows folder picker dialog, allowing you to select any folder on disk. The selected path is written directly into the adapter folder input.

---

## Upstream Features

All features from the upstream [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) are preserved:

- Complete style search with 936 styles synchronized from Suno's explorer
- Song parameter history — reuse any previous generation's settings
- Four-language localization (English, Chinese, Japanese, Korean)
- LoRA and LoKr training support with memory offloading optimization
- Basic single-adapter LoRA/LoKr loading

---

## 🔧 Setting up the Environment for Windows

Give unrestricted script access to PowerShell so venv can work:

- Open an administrator PowerShell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin PowerShell window

## Installation

Clone the repo with `--recurse-submodules`:

```
git clone --recurse-submodules https://github.com/scragnog/ACE-Step-1.5-for-windows.git -b qinglong
```

> ⚠️ **MUST USE `--recurse-submodules`** — the UI is a git submodule.

### Install Dependencies

Run the following PowerShell script:
```powershell
./1、install-uv-qinglong.ps1
```

### (Optional) VS Studio 2022 for torch compile
Download from Microsoft official link:
https://aka.ms/vs/17/release/vs_community.exe

Install C++ desktop and language package with English (especially for Asian computers).

### FFMPEG

https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-win64-gpl-shared-8.0.zip

Use the shared version for ffmpeg.

### Change Default Model
Copy `.env.sample` and rename to `.env`, then change the model name to your preference.

### Linux
1. First install PowerShell:
```bash
./0、install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1、install-uv-qinglong.ps1
```
Use `sudo pwsh` if you are on Linux without root user.

## Usage

### Option A: One-Click Launcher (Recommended)

Double-click **`LAUNCH.bat`** — this will:

1. Open an interactive loading screen with model selection dropdowns
2. Install UI dependencies if needed
3. Start the Express backend, then the Python API server
4. Auto-redirect to the app once all services are ready

The loading screen auto-populates model dropdowns from your `checkpoints/` folder and lets you change the startup models before the Python API loads them. If you don’t interact, it auto-continues after 5 seconds.

> **Alternative:** `START.bat` does the same thing without the loading screen — it opens three separate command windows and launches the browser directly after a short delay.

### Option B: Manual Launch (PowerShell Scripts)

If you prefer to start services independently:

```powershell
# Terminal 1 — Start the Python API backend
3、run_server.ps1

# Terminal 2 — Start the UI (Express + Vite frontend)
4、run_npmgui.ps1
```

Then open http://localhost:3000 in your browser.

---

## Credits

- **ACE-Step 1.5** — [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) (original model & backend)
- **Windows integration** — [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) (upstream fork)
- **Frontend** — [fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui) (original UI)
- **Audio Enhancement** — [ShmuelRonen/ComfyUI-Audio_Quality_Enhancer](https://github.com/ShmuelRonen/ComfyUI-Audio_Quality_Enhancer) (DSP engine inspiration)
