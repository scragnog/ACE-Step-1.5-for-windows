# ACE-Step Audio Quality Enhancement Options

Research notes from investigating approaches to improve ACE-Step's 25Hz VAE audio quality.

## Root Cause

ACE-Step's `AutoencoderOobleck` VAE has a 1920× downsampling ratio (strides `[2,4,4,6,10]`) at 48kHz, producing latents at **25Hz**. This causes:
- Phase incoherence → "fizzy" / metallic artifacts
- High-frequency smearing → loss of transient detail
- Spectral artifacts from coarse temporal resolution

## Tested Approaches

### ❌ EAR_VAE Post-Processing (audio → encode → decode)
- **Result**: More muffled, smoothed high end. No improvement.
- **Why**: Another VAE encode→decode is another lossy compression pass. EAR_VAE was trained to reconstruct from *its own* latents, not to fix another VAE's artifacts. The 960× bottleneck just smears detail further.
- Tested with: deterministic encoding (`--no-sample`), various blend ratios, v2 48kHz model.

### ❌ BigVGAN Mel-Spectrogram Re-Vocoding
- **Result**: "Interesting" but not better. Different quality character but not an improvement.
- **Why**: Even though BigVGAN reconstructs clean phase from mel magnitudes, the mel spectrogram itself (at 86Hz or 172Hz) loses too much fine temporal detail. The 48→44.1→48kHz resample round-trip also degrades quality.
- Tested with: 512x model (86Hz mel), 256x model (172Hz mel), blend variants, native 44.1kHz output.

### ❌ Direct VAE Swap (replace ACE-Step's VAE with EAR_VAE)
- **Feasibility**: Not possible without retraining the entire DiT + 5Hz LM stack.
- **Why**: Latent dimensions match (both 64-dim) but temporal resolution differs (25Hz vs ~47Hz) and learned distributions are incompatible.

## Untested Approaches

### 🥇 Traditional DSP Chain (No ML)
No ML — just signal processing. The "fizzy" ACE-Step sound has known spectral characteristics:
- Harmonic exciter (adds harmonics rather than boosting existing ones)
- Multi-band transient shaper (restore transient punch)
- Stereo imager + mid-side processing (fix phase coherence)
- Narrow-band notch filters on resonant peaks
- Could automate with `scipy` or `pedalboard`

### 🥈 Latent-Space Temporal Super-Resolution
- Save ACE-Step's 25Hz latents before VAE decode
- Train a small upsampler (25Hz → 50Hz latent frames)
- Decode upsampled latents with a higher-quality decoder
- Requires training data + GPU compute
- **This is the "right" fix** — addresses the root cause rather than post-processing

### 🥉 Diffusion-Based Audio Enhancement (SDEdit)
- Encode ACE-Step output to a different audio diffusion model's latent space
- Add partial noise (~30% steps), then denoise
- Diffusion model "fills in" high-frequency detail
- Risk: could alter musical content

## Key Lesson

Post-processing approaches (VAE roundtrip, mel-spectrogram vocoding) cannot meaningfully fix artifacts that stem from information loss during ACE-Step's 25Hz latent encoding. The temporal resolution ceiling is too fundamental. The solutions that could work require either:
1. **Modifying the generation pipeline** (latent super-resolution before decoding)
2. **Traditional DSP** (targeted spectral processing that doesn't discard existing information)
3. **Upstream fix** (ACE-Step team upgrading to a higher-resolution VAE)
