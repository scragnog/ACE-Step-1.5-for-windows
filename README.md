# ACE-Step 1.5 for Windows — Enhanced Fork

An enhanced fork of [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows) with a rebuilt UI experience, multi-adapter support, and quality-of-life improvements for music generation with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).

<img width="2062" height="952" alt="image" src="https://github.com/user-attachments/assets/6e682194-99f2-4267-b412-1b5198720b87" />

---

## ✨ New Features

> Full details and implementation notes in [FEATURES.md](FEATURES.md).

### 🧠 Advanced Guidance & Solvers
Total control over the generation pipeline with 7 unique mathematical guidance modes (APG, ADG, PAG, Plain CFG, CFG++, Dynamic CFG, Rescaled CFG) and 4 ODE solver algorithms (Euler, Heun, DPM++ 2M, RK4). Includes 40+ multilingual educational tooltips explaining every generation parameter.

### 🎯 Activation Steering (TADA)
> ⚠️ **Experimental Feature:** Currently in-progress and may not work as intended.

Guide generation without expanding prompt length using isolated internal brain state direction vectors. Includes a UI component to compute steering vectors for multiple concepts at once, override base genres, fine-tune strength (alpha) and model layer targets on-the-fly, and remove vectors directly from disk. See the [Activation Steering Tutorial](docs/en/Activation_Steering_Tutorial.md).

### 🎛️ Advanced Multi-Adapter System
Load up to **4 LoRA/LoKr adapters simultaneously** with independent per-slot scale sliders and per-module-group scaling (Self-Attn, Cross-Attn, MLP). Uses weight-space merging for zero-hook inference. Per-adapter settings persist across sessions. Includes a built-in **file browser** for scanning and loading `.safetensors` files from a configurable folder.

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
