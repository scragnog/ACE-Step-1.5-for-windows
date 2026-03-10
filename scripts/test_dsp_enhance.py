#!/usr/bin/env python
"""
Per-Stem DSP Enhancement Proof of Concept
==========================================
Splits ACE-Step audio using BS-RoFormer (vocals) + htdemucs_6s (instrumental),
applies mastering-grade DSP per stem, then recombines.

Designed to address ACE-Step's specific artifacts:
  - Vocals: fizzy sibilance, metallic resonance, thin body
  - Drums: smeared transients, muddy kick, thin snare
  - Bass: muddy sub, unclear harmonics
  - Other: thin, narrow stereo, fizzy top end

Usage:
    python scripts/test_dsp_enhance.py "path/to/song.wav"
    python scripts/test_dsp_enhance.py --preset aggressive "path/to/song.wav"
    python scripts/test_dsp_enhance.py --save-stems "path/to/song.wav"

Outputs saved alongside original with '_enhanced' suffix.
"""

import argparse
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

# Pedalboard for high-quality DSP
from pedalboard import (
    Pedalboard, Compressor, LowShelfFilter, HighShelfFilter,
    PeakFilter, Gain, Limiter, HighpassFilter, LowpassFilter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# BFloat16 workaround (same as stem_service.py)
# ---------------------------------------------------------------------------
@contextmanager
def _float32_default_dtype():
    try:
        import torch
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            yield
        finally:
            torch.set_default_dtype(prev)
    except ImportError:
        yield


# ---------------------------------------------------------------------------
# Stem Separation (RoFormer + htdemucs_6s two-pass)
# ---------------------------------------------------------------------------

ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEMUCS_6S_MODEL = "htdemucs_6s.yaml"


def separate_stems(audio_path: str, output_dir: str) -> Tuple[Dict[str, np.ndarray], int]:
    """Two-pass stem separation: RoFormer vocals + htdemucs_6s instrumental.

    Returns (stems_dict, stem_sample_rate) where stems_dict maps
    stem_name -> numpy array [channels, samples].
    """
    from audio_separator.separator import Separator

    print("  Loading audio_separator...")
    sep = Separator()
    sep.output_format = "wav"

    stems = {}
    stem_sr = None

    with _float32_default_dtype():
        # Pass 1: BS-RoFormer → vocals + instrumental
        print(f"  Pass 1/2: Isolating vocals with BS-RoFormer...")
        sep.output_dir = output_dir
        sep.load_model(model_filename=ROFORMER_MODEL)
        pass1_files = sep.separate(audio_path)

        vocals_path = None
        instrumental_path = None
        for fp in pass1_files:
            fp = str(Path(fp) if Path(fp).is_absolute() else Path(output_dir) / fp)
            fname = Path(fp).stem.lower()
            if "vocal" in fname and "instrument" not in fname:
                vocals_path = fp
            else:
                instrumental_path = fp

        if vocals_path:
            data, stem_sr = sf.read(vocals_path, dtype="float32")
            stems["vocals"] = data.T if data.ndim == 2 else data.reshape(1, -1)
            print(f"    Vocals: {stems['vocals'].shape}, {stem_sr}Hz")

        if not instrumental_path:
            print("  WARNING: No instrumental found, returning vocals only")
            return stems, stem_sr or 44100

        # Pass 2: htdemucs_6s on instrumental → drums, bass, guitar, piano, other
        print(f"  Pass 2/2: Splitting instrumental with htdemucs_6s...")
        pass2_dir = os.path.join(output_dir, "pass2")
        os.makedirs(pass2_dir, exist_ok=True)
        sep.output_dir = pass2_dir
        sep.load_model(model_filename=DEMUCS_6S_MODEL)
        pass2_files = sep.separate(instrumental_path)

        for fp in pass2_files:
            fp = str(Path(fp) if Path(fp).is_absolute() else Path(pass2_dir) / fp)
            fname = Path(fp).stem.lower()
            if "vocal" in fname:
                continue  # skip — we have RoFormer vocals

            # Classify stem type
            stem_type = "other"
            for candidate in ("drums", "bass", "guitar", "piano"):
                if candidate in fname:
                    stem_type = candidate
                    break

            data, file_sr = sf.read(fp, dtype="float32")
            if stem_sr is None:
                stem_sr = file_sr
            stems[stem_type] = data.T if data.ndim == 2 else data.reshape(1, -1)
            print(f"    {stem_type}: {stems[stem_type].shape}, {file_sr}Hz")

    return stems, stem_sr or 44100


# ---------------------------------------------------------------------------
# Adaptive Spectral Analysis
# ---------------------------------------------------------------------------

def analyze_stem(audio: np.ndarray, sr: int, name: str = "") -> Dict[str, float]:
    """Analyze a stem's spectral characteristics to drive adaptive DSP.

    Returns a dict of metrics (0-1 normalized) that describe the audio's
    existing qualities. Enhancement functions use these to scale processing:
    - If warmth is already high, reduce warmth boost
    - If harshness is high, increase de-harsh
    - If transients are already sharp, reduce transient shaping
    """
    # Work with mono sum for analysis
    mono = np.mean(audio, axis=0) if audio.ndim > 1 and audio.shape[0] > 1 else audio.flatten()

    # Skip near-silent stems
    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 1e-6:
        return {"warmth": 0.5, "brightness": 0.5, "harshness": 0.5,
                "dynamic_range": 0.5, "transient_sharpness": 0.5, "rms": 0.0}

    # Compute power spectrum
    n_fft = min(4096, len(mono))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Average over multiple windows for stability
    n_windows = min(20, max(1, len(mono) // n_fft))
    hop = max(1, (len(mono) - n_fft) // n_windows)
    power_sum = np.zeros(len(freqs))
    for i in range(n_windows):
        start = i * hop
        chunk = mono[start:start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(n_fft))) ** 2
        power_sum += spectrum
    power = power_sum / n_windows
    power_db = 10 * np.log10(power + 1e-10)

    # Band energy ratios
    def band_energy(low_hz, high_hz):
        mask = (freqs >= low_hz) & (freqs < high_hz)
        return np.mean(power[mask]) if np.any(mask) else 0.0

    total_energy = np.mean(power) + 1e-10
    sub_bass = band_energy(20, 100) / total_energy
    low_mid = band_energy(100, 500) / total_energy
    mid = band_energy(500, 2000) / total_energy
    upper_mid = band_energy(2000, 5000) / total_energy
    harsh_zone = band_energy(4000, 8000) / total_energy
    air = band_energy(8000, 20000) / total_energy

    # Warmth: ratio of low-mid energy to total (0-1, higher = warmer)
    warmth = np.clip((sub_bass + low_mid) * 3.0, 0, 1)

    # Brightness: ratio of upper frequencies to total
    brightness = np.clip((upper_mid + air) * 4.0, 0, 1)

    # Harshness: energy in the 4-8kHz "fizz zone" relative to neighbors
    neighbor_energy = (band_energy(2000, 4000) + band_energy(8000, 12000)) / 2
    harsh_ratio = band_energy(4000, 8000) / (neighbor_energy + 1e-10)
    harshness = np.clip((harsh_ratio - 0.5) * 2.0, 0, 1)

    # Dynamic range: ratio of peak to RMS (crest factor)
    peak = np.max(np.abs(mono))
    crest = peak / (rms + 1e-10)
    dynamic_range = np.clip((crest - 1.0) / 15.0, 0, 1)  # normalize ~1-16 range

    # Transient sharpness: how fast envelope rises
    env = np.abs(mono)
    win = max(int(0.005 * sr), 1)  # 5ms window
    smoothed = np.convolve(env, np.ones(win) / win, mode='same')
    env_diff = np.diff(smoothed)
    transient_sharpness = np.clip(np.percentile(env_diff[env_diff > 0], 95) * 20, 0, 1)

    profile = {
        "warmth": float(warmth),
        "brightness": float(brightness),
        "harshness": float(harshness),
        "dynamic_range": float(dynamic_range),
        "transient_sharpness": float(transient_sharpness),
        "rms": float(rms),
    }

    if name:
        print(f"    [{name}] Analysis: " + ", ".join(f"{k}={v:.2f}" for k, v in profile.items()))

    return profile


def _adaptive_scale(base_value: float, metric: float, target: float = 0.5,
                    sensitivity: float = 1.0) -> float:
    """Scale a DSP parameter based on how far a metric is from its target.

    For BOOSTS (positive base): metric > target → less boost needed
    For CUTS (negative base): metric > target → MORE cut needed
    """
    deficit = target - metric  # positive = needs more of this quality
    if base_value < 0:
        # For cuts: invert logic — high metric means we need more cutting
        deficit = metric - target
    scale = 1.0 + deficit * sensitivity
    return base_value * np.clip(scale, 0.2, 2.0)


# ---------------------------------------------------------------------------
# DSP Processing Chains
# ---------------------------------------------------------------------------

def _saturate(audio: np.ndarray, drive: float = 1.5) -> np.ndarray:
    """Soft-clip saturation using tanh. drive > 1.0 adds harmonics."""
    if drive <= 1.0:
        return audio
    return np.tanh(audio * drive) / np.tanh(drive)


def _transient_shape(audio: np.ndarray, sr: int,
                     attack_boost: float = 0.4,
                     sustain_cut: float = 0.0,
                     attack_ms: float = 5.0,
                     release_ms: float = 50.0) -> np.ndarray:
    """Proper transient shaper using envelope follower with attack/release.

    attack_boost: how much to boost transient attacks (0-1)
    sustain_cut: how much to reduce sustain (0-1, negative = boost)
    """
    result = audio.copy()
    attack_samples = max(int(attack_ms / 1000 * sr), 1)
    release_samples = max(int(release_ms / 1000 * sr), 1)

    for ch in range(result.shape[0]):
        # Compute slow and fast envelopes
        fast_env = np.zeros(result.shape[1])
        slow_env = np.zeros(result.shape[1])
        abs_signal = np.abs(result[ch])

        # Fast follower (transients)
        fast_attack = 1.0 - np.exp(-2.2 / attack_samples)
        fast_release = 1.0 - np.exp(-2.2 / (release_samples * 2))
        for i in range(1, len(abs_signal)):
            if abs_signal[i] > fast_env[i-1]:
                fast_env[i] = fast_attack * abs_signal[i] + (1 - fast_attack) * fast_env[i-1]
            else:
                fast_env[i] = fast_release * abs_signal[i] + (1 - fast_release) * fast_env[i-1]

        # Slow follower (sustain)
        slow_attack = 1.0 - np.exp(-2.2 / (attack_samples * 10))
        slow_release = 1.0 - np.exp(-2.2 / (release_samples * 10))
        for i in range(1, len(abs_signal)):
            if abs_signal[i] > slow_env[i-1]:
                slow_env[i] = slow_attack * abs_signal[i] + (1 - slow_attack) * slow_env[i-1]
            else:
                slow_env[i] = slow_release * abs_signal[i] + (1 - slow_release) * slow_env[i-1]

        # Transient = fast - slow (positive where transients are)
        transient = fast_env - slow_env
        transient = np.clip(transient, 0, None)

        # Sustain = where slow > some threshold
        sustain_mask = slow_env / (np.max(slow_env) + 1e-8)

        # Apply shaping
        gain = np.ones(result.shape[1])
        if attack_boost > 0:
            gain += transient * attack_boost * 10  # scale factor for audibility
        if sustain_cut > 0:
            gain -= sustain_mask * sustain_cut * 0.5

        gain = np.clip(gain, 0.3, 3.0)  # safety limits
        result[ch] *= gain

    return result


def enhance_vocals(audio: np.ndarray, sr: int, intensity: float = 1.0,
                   profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive vocal processing. Adjusts based on spectral analysis."""
    p = profile or {}

    # Adaptive scaling
    dehash_db = _adaptive_scale(-3.0 * intensity, p.get("harshness", 0.5), target=0.3, sensitivity=1.5)
    warmth_db = _adaptive_scale(1.5 * intensity, p.get("warmth", 0.5), target=0.6, sensitivity=1.5)
    presence_db = _adaptive_scale(2.0 * intensity, p.get("brightness", 0.5), target=0.5, sensitivity=1.0)
    air_db = _adaptive_scale(2.0 * intensity, p.get("brightness", 0.5), target=0.4, sensitivity=1.0)
    comp_ratio = _adaptive_scale(2.5, p.get("dynamic_range", 0.5), target=0.4, sensitivity=1.0)

    board = Pedalboard([
        # De-harsh: surgical cuts in the fizz zone (tight Q = narrow band)
        PeakFilter(cutoff_frequency_hz=6000, gain_db=dehash_db, q=3.0),
        PeakFilter(cutoff_frequency_hz=8000, gain_db=dehash_db * 0.6, q=3.0),

        # Body/warmth: gentle boost
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=warmth_db, q=0.7),

        # Presence: clarity below the fizz zone
        PeakFilter(cutoff_frequency_hz=2500, gain_db=presence_db, q=1.0),

        # Compression: moderate, EQ-aware
        Compressor(
            threshold_db=-16, ratio=max(1.5, comp_ratio),
            attack_ms=10.0, release_ms=100.0
        ),

        # Air: sparkle above the fizz zone
        HighShelfFilter(cutoff_frequency_hz=12000, gain_db=air_db, q=0.7),

        Gain(gain_db=1.0 * intensity),
    ])

    result = audio.copy()
    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)

    # Saturation: less if already warm
    sat_drive = 1.0 + _adaptive_scale(0.2 * intensity, p.get("warmth", 0.5), target=0.6, sensitivity=1.0)
    result = _saturate(result, drive=sat_drive)

    return result


def enhance_drums(audio: np.ndarray, sr: int, intensity: float = 1.0,
                  profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive drum processing."""
    p = profile or {}

    # Transient shaping: less if transients are already sharp
    atk_boost = _adaptive_scale(0.3 * intensity, p.get("transient_sharpness", 0.3), target=0.6, sensitivity=1.5)
    result = _transient_shape(
        audio, sr,
        attack_boost=atk_boost,
        sustain_cut=0.05 * intensity,
        attack_ms=3.0,
        release_ms=30.0,
    )

    kick_db = _adaptive_scale(2.5 * intensity, p.get("warmth", 0.5), target=0.5, sensitivity=1.0)
    snap_db = _adaptive_scale(2.0 * intensity, p.get("brightness", 0.5), target=0.5, sensitivity=1.0)
    air_db = _adaptive_scale(1.5 * intensity, p.get("brightness", 0.5), target=0.4, sensitivity=1.0)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        # Kick punch at sub level
        PeakFilter(cutoff_frequency_hz=70, gain_db=kick_db, q=1.2),
        # Mud cut higher up (400Hz, away from kick fundamental)
        PeakFilter(cutoff_frequency_hz=400, gain_db=-1.5 * intensity, q=0.8),
        # Snare snap
        PeakFilter(cutoff_frequency_hz=3000, gain_db=snap_db, q=1.0),
        # Cymbal air
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air_db, q=0.7),
        # Moderate compression (let transients breathe)
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=5.0, release_ms=50.0),
        Gain(gain_db=1.5 * intensity),
    ])

    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)
    return result


def enhance_bass(audio: np.ndarray, sr: int, intensity: float = 1.0,
                 profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive bass processing."""
    p = profile or {}
    result = audio.copy()

    # Mono below 100Hz (always — tightens sub-bass)
    if result.shape[0] >= 2:
        nyquist = sr / 2
        low_cut = min(100 / nyquist, 0.99)
        lp_sos = signal.butter(4, low_cut, btype='lowpass', output='sos')
        hp_sos = signal.butter(4, low_cut, btype='highpass', output='sos')
        sub_l = signal.sosfilt(lp_sos, result[0])
        sub_r = signal.sosfilt(lp_sos, result[1])
        upper_l = signal.sosfilt(hp_sos, result[0])
        upper_r = signal.sosfilt(hp_sos, result[1])
        sub_mono = (sub_l + sub_r) * 0.5
        result[0] = sub_mono + upper_l
        result[1] = sub_mono + upper_r

    sub_db = _adaptive_scale(2.5 * intensity, p.get("warmth", 0.5), target=0.6, sensitivity=1.0)
    harmonic_db = _adaptive_scale(2.0 * intensity, p.get("brightness", 0.3), target=0.4, sensitivity=1.5)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        PeakFilter(cutoff_frequency_hz=60, gain_db=sub_db, q=1.0),
        PeakFilter(cutoff_frequency_hz=250, gain_db=-2.0 * intensity, q=0.8),
        PeakFilter(cutoff_frequency_hz=700, gain_db=harmonic_db, q=1.0),
        Compressor(threshold_db=-16, ratio=3.5, attack_ms=5.0, release_ms=60.0),
        LowpassFilter(cutoff_frequency_hz=5000),
        Gain(gain_db=1.0 * intensity),
    ])

    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)

    sat_drive = 1.0 + _adaptive_scale(0.5 * intensity, p.get("warmth", 0.5), target=0.5, sensitivity=1.0)
    result = _saturate(result, drive=sat_drive)
    return result


def enhance_guitar(audio: np.ndarray, sr: int, intensity: float = 1.0,
                   profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive guitar enhancement."""
    p = profile or {}
    defizz_db = _adaptive_scale(-2.0 * intensity, p.get("harshness", 0.5), target=0.3, sensitivity=2.0)
    body_db = _adaptive_scale(1.5 * intensity, p.get("warmth", 0.5), target=0.5, sensitivity=1.0)

    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=500, gain_db=body_db, q=0.8),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0 * intensity, q=1.0),
        PeakFilter(cutoff_frequency_hz=6000, gain_db=defizz_db, q=1.5),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5 * intensity, q=0.7),
        Compressor(threshold_db=-18, ratio=2.5, attack_ms=10.0, release_ms=80.0),
        Gain(gain_db=1.0 * intensity),
    ])

    result = audio.copy()
    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)
    result = _saturate(result, drive=1.0 + 0.2 * intensity)
    return result


def enhance_piano(audio: np.ndarray, sr: int, intensity: float = 1.0,
                  profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive piano enhancement."""
    p = profile or {}
    warmth_db = _adaptive_scale(1.5 * intensity, p.get("warmth", 0.5), target=0.5, sensitivity=1.0)
    defizz_db = _adaptive_scale(-2.0 * intensity, p.get("harshness", 0.5), target=0.3, sensitivity=2.0)

    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=warmth_db, q=0.8),
        PeakFilter(cutoff_frequency_hz=2500, gain_db=1.5 * intensity, q=1.0),
        PeakFilter(cutoff_frequency_hz=5500, gain_db=defizz_db, q=1.5),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0 * intensity, q=0.7),
        Compressor(threshold_db=-16, ratio=2.0, attack_ms=15.0, release_ms=100.0),
        Gain(gain_db=0.5 * intensity),
    ])

    result = audio.copy()
    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)
    return result


def enhance_other(audio: np.ndarray, sr: int, intensity: float = 1.0,
                  profile: Dict[str, float] = None) -> np.ndarray:
    """Adaptive generic instrument enhancement."""
    p = profile or {}
    body_db = _adaptive_scale(1.5 * intensity, p.get("warmth", 0.5), target=0.5, sensitivity=1.0)
    defizz_db = _adaptive_scale(-2.0 * intensity, p.get("harshness", 0.5), target=0.3, sensitivity=2.0)
    presence_db = _adaptive_scale(2.0 * intensity, p.get("brightness", 0.5), target=0.5, sensitivity=1.0)

    board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=body_db, q=0.7),
        PeakFilter(cutoff_frequency_hz=2000, gain_db=presence_db, q=1.0),
        PeakFilter(cutoff_frequency_hz=6000, gain_db=defizz_db, q=1.5),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5 * intensity, q=0.7),
        Compressor(threshold_db=-18, ratio=2.0, attack_ms=10.0, release_ms=80.0),
        Gain(gain_db=1.0 * intensity),
    ])

    result = audio.copy()
    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sr)
    return result


# ---------------------------------------------------------------------------
# Stereo Enhancement (applied to final mix)
# ---------------------------------------------------------------------------

def enhance_stereo(audio: np.ndarray, sr: int, width: float = 0.3) -> np.ndarray:
    """Mid-side stereo enhancement. width 0-1."""
    if audio.shape[0] < 2 or width <= 0:
        return audio

    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5

    # Only widen above 200Hz (keep bass centered)
    nyquist = sr / 2
    hp_freq = min(200 / nyquist, 0.99)
    hp_sos = signal.butter(2, hp_freq, btype='highpass', output='sos')
    side_filtered = signal.sosfilt(hp_sos, side)

    # Boost side channel
    side_boosted = side_filtered * (1.0 + width * 1.5)

    result = np.vstack([
        mid + side_boosted,
        mid - side_boosted,
    ])
    return result


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "subtle": {
        "label": "Subtle (light touch)",
        "intensity": 0.5,
        "stereo_width": 0.1,
    },
    "balanced": {
        "label": "Balanced (recommended)",
        "intensity": 1.0,
        "stereo_width": 0.2,
    },
    "aggressive": {
        "label": "Aggressive (maximum enhancement)",
        "intensity": 1.5,
        "stereo_width": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_audio(input_path: Path, output_path: Path,
                  intensity: float = 1.0, stereo_width: float = 0.2,
                  save_stems: bool = False):
    """Full pipeline: separate → enhance per stem → remix → master."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"  Intensity: {intensity:.1f}, Stereo width: {stereo_width:.1f}")
    t_start = time.time()

    # Load original for reference
    orig_data, sr = sf.read(str(input_path), dtype="float32")
    if orig_data.ndim == 1:
        orig_data = np.stack([orig_data, orig_data], axis=-1)
    print(f"  Input: {orig_data.shape}, {sr}Hz, {orig_data.shape[0]/sr:.1f}s")

    # Separate stems
    print(f"\n--- Stem Separation ---")
    with tempfile.TemporaryDirectory(prefix="ace_dsp_") as tmp_dir:
        stems, stem_sr = separate_stems(str(input_path), tmp_dir)

    if not stems:
        print("  ERROR: No stems extracted!")
        return False

    # Resample stems to match original sample rate if needed
    if stem_sr != sr:
        import librosa
        print(f"  Resampling stems from {stem_sr}Hz to {sr}Hz...")
        for name in stems:
            resampled_channels = []
            for ch in range(stems[name].shape[0]):
                resampled_channels.append(
                    librosa.resample(stems[name][ch], orig_sr=stem_sr, target_sr=sr)
                )
            stems[name] = np.stack(resampled_channels)
        print(f"  Resampled all {len(stems)} stems")

    print(f"\n--- Spectral Analysis ---")
    profiles = {}
    for name, data in stems.items():
        profiles[name] = analyze_stem(data, sr, name)

    print(f"\n--- Per-Stem Enhancement (adaptive) ---")

    enhancers = {
        "vocals": enhance_vocals,
        "drums": enhance_drums,
        "bass": enhance_bass,
        "guitar": enhance_guitar,
        "piano": enhance_piano,
        "other": enhance_other,
    }

    enhanced_stems = {}
    for name, data in stems.items():
        enhancer = enhancers.get(name, enhance_other)
        print(f"  Enhancing {name}...")
        enhanced_stems[name] = enhancer(data, sr, intensity, profile=profiles.get(name))

    # Save individual stems if requested
    if save_stems:
        stem_dir = output_path.parent / f"{input_path.stem}_stems"
        stem_dir.mkdir(exist_ok=True)
        for name, data in enhanced_stems.items():
            stem_path = stem_dir / f"{name}_enhanced.wav"
            sf.write(str(stem_path), data.T, sr, subtype="FLOAT")
            print(f"  Saved stem: {stem_path.name}")
        # Also save originals for comparison
        for name, data in stems.items():
            stem_path = stem_dir / f"{name}_original.wav"
            sf.write(str(stem_path), data.T, sr, subtype="FLOAT")

    # Remix
    print(f"\n--- Remix & Master ---")
    max_len = max(s.shape[1] for s in enhanced_stems.values())
    num_ch = max(s.shape[0] for s in enhanced_stems.values())
    remix = np.zeros((num_ch, max_len))

    for name, data in enhanced_stems.items():
        if data.shape[0] < num_ch:
            data = np.vstack([data] * num_ch)[:num_ch]
        remix[:, :data.shape[1]] += data

    # Stereo enhancement on final mix
    if stereo_width > 0 and remix.shape[0] >= 2:
        print(f"  Applying stereo enhancement (width={stereo_width:.1f})...")
        remix = enhance_stereo(remix, sr, stereo_width)

    # Final limiter
    print(f"  Applying final limiter...")
    limiter = Pedalboard([
        Limiter(threshold_db=-0.5, release_ms=50.0)
    ])
    for ch in range(remix.shape[0]):
        max_val = np.max(np.abs(remix[ch]))
        if max_val > 1.0:
            remix[ch] = remix[ch] / max_val
        remix[ch] = limiter.process(remix[ch], sample_rate=sr)

    # Save
    print(f"  Output: {remix.shape}, saving to: {output_path.name}")
    sf.write(str(output_path), remix.T, sr, subtype="FLOAT")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Per-stem DSP enhancement for ACE-Step audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", help="Input WAV file(s)")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="balanced",
                        help="Enhancement preset (default: balanced)")
    parser.add_argument("--intensity", type=float, default=None,
                        help="Override preset intensity (0.5-2.0)")
    parser.add_argument("--stereo-width", type=float, default=None,
                        help="Override stereo width (0.0-0.5)")
    parser.add_argument("--save-stems", action="store_true",
                        help="Save individual stems (before/after) for comparison")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    preset = PRESETS[args.preset]
    intensity = args.intensity if args.intensity is not None else preset["intensity"]
    stereo_width = args.stereo_width if args.stereo_width is not None else preset["stereo_width"]

    print(f"Preset: {preset['label']}")
    print(f"Intensity: {intensity}, Stereo width: {stereo_width}")

    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"Warning: Skipping {f} (not found)")
            continue

        suffix = f"_enhanced"
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{p.stem}{suffix}.wav"
        else:
            output_path = p.parent / f"{p.stem}{suffix}.wav"

        try:
            process_audio(p, output_path, intensity, stereo_width, args.save_stems)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done! Compare original and _enhanced files in your audio player/DAW.")


if __name__ == "__main__":
    main()
