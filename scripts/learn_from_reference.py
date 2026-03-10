#!/usr/bin/env python
"""
Learn DSP Profile from Reference
=================================
Compares an original audio file to an Ozone-processed version and extracts
the DSP transfer function. Outputs a JSON profile that can be applied to
other files using pedalboard.

Usage:
    python scripts/learn_from_reference.py original.wav processed.wav
    python scripts/learn_from_reference.py original.wav processed.wav --apply target.wav

Pipeline:
  1. Load both files, align lengths
  2. Extract spectral transfer function (what EQ was applied)
  3. Fit parametric EQ bands to approximate the curve
  4. Estimate compression, stereo width, and saturation changes
  5. Output JSON profile + optional PNG visualization
  6. Optionally apply the learned profile to a new file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal, optimize

from pedalboard import (
    Pedalboard, Compressor, LowShelfFilter, HighShelfFilter,
    PeakFilter, Gain, Limiter, HighpassFilter, LowpassFilter,
)


# ---------------------------------------------------------------------------
# Transfer Function Extraction
# ---------------------------------------------------------------------------

def extract_transfer_function(
    original: np.ndarray, processed: np.ndarray, sr: int,
    n_fft: int = 8192, smoothing_octaves: float = 1/3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frequency response (transfer function) between two signals.

    Returns (frequencies, magnitude_db).
    """
    # Work with mono sum
    if original.ndim > 1 and original.shape[0] > 1:
        orig_mono = np.mean(original, axis=0)
        proc_mono = np.mean(processed, axis=0)
    else:
        orig_mono = original.flatten()
        proc_mono = processed.flatten()

    # Align lengths
    min_len = min(len(orig_mono), len(proc_mono))
    orig_mono = orig_mono[:min_len]
    proc_mono = proc_mono[:min_len]

    # Compute averaged power spectra (Welch's method)
    freqs, psd_orig = signal.welch(orig_mono, sr, nperseg=n_fft, noverlap=n_fft//2)
    _, psd_proc = signal.welch(proc_mono, sr, nperseg=n_fft, noverlap=n_fft//2)

    # Transfer function = processed / original (in power)
    # Avoid division by zero
    mask = psd_orig > 1e-20
    tf = np.ones_like(psd_orig)
    tf[mask] = psd_proc[mask] / psd_orig[mask]

    # Convert to dB
    tf_db = 10 * np.log10(tf + 1e-20)

    # Smooth with fractional-octave averaging
    tf_db_smooth = _smooth_octave(freqs, tf_db, smoothing_octaves)

    # Separate overall gain from EQ shape
    # The Maximizer adds a roughly flat loudness boost across all frequencies.
    # Subtract the median to isolate the actual EQ curve.
    audible_mask = (freqs >= 100) & (freqs <= 15000)
    if np.any(audible_mask):
        overall_gain_db = float(np.median(tf_db_smooth[audible_mask]))
    else:
        overall_gain_db = float(np.median(tf_db_smooth))

    eq_shape_db = tf_db_smooth - overall_gain_db

    return freqs, eq_shape_db, overall_gain_db


def _smooth_octave(freqs: np.ndarray, values: np.ndarray,
                   octave_fraction: float = 1/3) -> np.ndarray:
    """Fractional-octave smoothing of a frequency-domain signal."""
    result = values.copy()
    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        f_low = f / (2 ** (octave_fraction / 2))
        f_high = f * (2 ** (octave_fraction / 2))
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask):
            result[i] = np.mean(values[mask])
    return result


# ---------------------------------------------------------------------------
# Parametric EQ Band Fitting
# ---------------------------------------------------------------------------

def _peak_filter_response(freqs: np.ndarray, sr: int,
                          fc: float, gain_db: float, q: float) -> np.ndarray:
    """Compute the magnitude response (dB) of a parametric peak filter."""
    if abs(gain_db) < 0.01:
        return np.zeros_like(freqs, dtype=float)

    w0 = 2 * np.pi * fc / sr
    A = 10 ** (gain_db / 40)  # sqrt of linear gain
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]

    w, h = signal.freqz(b, a, worN=freqs, fs=sr)
    return 20 * np.log10(np.abs(h) + 1e-20)


def _shelf_response(freqs: np.ndarray, sr: int,
                    fc: float, gain_db: float, shelf_type: str) -> np.ndarray:
    """Compute magnitude response of a shelf filter."""
    if abs(gain_db) < 0.01:
        return np.zeros_like(freqs, dtype=float)

    # Use scipy to design shelf filter
    if shelf_type == "low":
        # Approximate low shelf as a 1st order IIR
        w0 = 2 * np.pi * fc / sr
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.7 - 1) + 2)

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)

        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:  # high shelf
        w0 = 2 * np.pi * fc / sr
        A = 10 ** (gain_db / 40)
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.7 - 1) + 2)

        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(A)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b = [b0/a0, b1/a0, b2/a0]
    a = [1, a1/a0, a2/a0]

    w, h = signal.freqz(b, a, worN=freqs, fs=sr)
    return 20 * np.log10(np.abs(h) + 1e-20)


def fit_eq_bands(
    freqs: np.ndarray, target_db: np.ndarray, sr: int,
    n_bands: int = 8,
) -> List[Dict]:
    """Fit parametric EQ bands to approximate a target frequency response.

    Returns list of band dicts: {type, freq_hz, gain_db, q}
    """
    # Focus on audible range
    audible = (freqs >= 20) & (freqs <= 20000)
    fit_freqs = freqs[audible]
    fit_target = target_db[audible]

    # Start with a low shelf, high shelf, and parametric bands
    # Initial band center frequencies spread logarithmically
    log_min = np.log10(60)
    log_max = np.log10(16000)
    initial_centers = np.logspace(log_min, log_max, n_bands - 2)

    bands = []

    # Low shelf
    bands.append({
        "type": "low_shelf",
        "freq_hz": 80.0,
        "gain_db": float(np.mean(fit_target[fit_freqs < 100])),
        "q": 0.7,
    })

    # Parametric bands
    for fc in initial_centers:
        # Get target gain at this frequency
        idx = np.argmin(np.abs(fit_freqs - fc))
        gain = float(fit_target[idx])
        bands.append({
            "type": "peak",
            "freq_hz": float(fc),
            "gain_db": gain,
            "q": 1.0,
        })

    # High shelf
    bands.append({
        "type": "high_shelf",
        "freq_hz": 10000.0,
        "gain_db": float(np.mean(fit_target[fit_freqs > 10000])),
        "q": 0.7,
    })

    # Iterative refinement: optimize each band to reduce residual error
    residual = fit_target.copy()
    optimized_bands = []

    for band in bands:
        if band["type"] == "peak":
            # Optimize freq, gain, and Q for this band
            def cost(params):
                fc, gain, q = params
                if fc < 20 or fc > 20000 or q < 0.1 or q > 10 or abs(gain) > 15:
                    return 1e6
                response = _peak_filter_response(fit_freqs, sr, fc, gain, q)
                new_residual = residual - response
                return np.sum(new_residual ** 2)

            try:
                result = optimize.minimize(
                    cost,
                    [band["freq_hz"], band["gain_db"], band["q"]],
                    method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1.0, "fatol": 0.01},
                )
                fc, gain, q = result.x
                band["freq_hz"] = float(np.clip(fc, 20, 20000))
                band["gain_db"] = float(np.clip(gain, -15, 15))
                band["q"] = float(np.clip(q, 0.1, 10))
            except Exception:
                pass

            response = _peak_filter_response(fit_freqs, sr, band["freq_hz"], band["gain_db"], band["q"])
        elif band["type"] == "low_shelf":
            response = _shelf_response(fit_freqs, sr, band["freq_hz"], band["gain_db"], "low")
        else:
            response = _shelf_response(fit_freqs, sr, band["freq_hz"], band["gain_db"], "high")

        residual -= response
        optimized_bands.append(band)

    # Filter out near-zero bands
    optimized_bands = [b for b in optimized_bands if abs(b["gain_db"]) > 0.1]

    return optimized_bands


# ---------------------------------------------------------------------------
# Dynamics Analysis
# ---------------------------------------------------------------------------

def analyze_dynamics(original: np.ndarray, processed: np.ndarray, sr: int) -> Dict:
    """Estimate compression parameters by comparing dynamic envelopes."""
    orig_mono = np.mean(original, axis=0) if original.ndim > 1 else original.flatten()
    proc_mono = np.mean(processed, axis=0) if processed.ndim > 1 else processed.flatten()

    min_len = min(len(orig_mono), len(proc_mono))
    orig_mono = orig_mono[:min_len]
    proc_mono = proc_mono[:min_len]

    # RMS envelopes (50ms windows)
    win = max(int(0.05 * sr), 1)
    hop = win // 2

    orig_rms = []
    proc_rms = []
    for i in range(0, min_len - win, hop):
        orig_rms.append(np.sqrt(np.mean(orig_mono[i:i+win] ** 2)))
        proc_rms.append(np.sqrt(np.mean(proc_mono[i:i+win] ** 2)))

    orig_rms = np.array(orig_rms)
    proc_rms = np.array(proc_rms)

    # Convert to dB
    orig_db = 20 * np.log10(orig_rms + 1e-10)
    proc_db = 20 * np.log10(proc_rms + 1e-10)

    # Gain reduction = processed - original (in dB)
    gain_reduction = proc_db - orig_db

    # Dynamic range comparison
    orig_dyn_range = float(np.percentile(orig_db, 95) - np.percentile(orig_db, 5))
    proc_dyn_range = float(np.percentile(proc_db, 95) - np.percentile(proc_db, 5))

    # Estimate compression ratio from dynamic range reduction
    if proc_dyn_range > 0 and orig_dyn_range > 0:
        estimated_ratio = orig_dyn_range / proc_dyn_range
    else:
        estimated_ratio = 1.0

    # Estimate threshold: where gain reduction starts
    # Find the loudness level where gain_reduction deviates from the median
    loud_mask = orig_db > np.percentile(orig_db, 50)
    if np.any(loud_mask):
        median_gr = float(np.median(gain_reduction))
        deviation = gain_reduction - median_gr
        threshold_candidates = orig_db[deviation < -1.0]
        if len(threshold_candidates) > 0:
            estimated_threshold = float(np.min(threshold_candidates))
        else:
            estimated_threshold = -12.0
    else:
        estimated_threshold = -12.0

    # Overall gain change
    gain_change = float(np.median(gain_reduction))

    return {
        "estimated_ratio": round(float(np.clip(estimated_ratio, 1.0, 10.0)), 2),
        "estimated_threshold_db": round(float(np.clip(estimated_threshold, -40, 0)), 1),
        "dynamic_range_original_db": round(orig_dyn_range, 1),
        "dynamic_range_processed_db": round(proc_dyn_range, 1),
        "gain_change_db": round(gain_change, 1),
    }


# ---------------------------------------------------------------------------
# Stereo Analysis
# ---------------------------------------------------------------------------

def analyze_stereo(original: np.ndarray, processed: np.ndarray, sr: int) -> Dict:
    """Compare stereo width between original and processed."""
    if original.shape[0] < 2 or processed.shape[0] < 2:
        return {"width_change": 0.0}

    min_len = min(original.shape[1], processed.shape[1])

    # Mid/side energy ratios
    def ms_ratio(audio):
        mid = (audio[0, :min_len] + audio[1, :min_len]) * 0.5
        side = (audio[0, :min_len] - audio[1, :min_len]) * 0.5
        mid_energy = np.mean(mid ** 2)
        side_energy = np.mean(side ** 2)
        return side_energy / (mid_energy + 1e-10)

    orig_ratio = ms_ratio(original)
    proc_ratio = ms_ratio(processed)

    width_change = float(proc_ratio / (orig_ratio + 1e-10) - 1.0)

    return {
        "width_change": round(width_change, 3),
        "original_side_ratio": round(float(orig_ratio), 4),
        "processed_side_ratio": round(float(proc_ratio), 4),
    }


# ---------------------------------------------------------------------------
# Saturation/Exciter Analysis
# ---------------------------------------------------------------------------

def analyze_exciter(original: np.ndarray, processed: np.ndarray, sr: int) -> Dict:
    """Estimate harmonic exciter settings by comparing harmonic content."""
    orig_mono = np.mean(original, axis=0) if original.ndim > 1 else original.flatten()
    proc_mono = np.mean(processed, axis=0) if processed.ndim > 1 else processed.flatten()
    min_len = min(len(orig_mono), len(proc_mono))

    # Compare THD (total harmonic distortion)
    n_fft = 8192
    _, psd_orig = signal.welch(orig_mono[:min_len], sr, nperseg=n_fft)
    freqs, psd_proc = signal.welch(proc_mono[:min_len], sr, nperseg=n_fft)

    # Look at harmonic energy above 5kHz relative to fundamental energy
    low_mask = (freqs >= 100) & (freqs < 5000)
    high_mask = (freqs >= 5000) & (freqs < 20000)

    orig_ratio = np.sum(psd_orig[high_mask]) / (np.sum(psd_orig[low_mask]) + 1e-10)
    proc_ratio = np.sum(psd_proc[high_mask]) / (np.sum(psd_proc[low_mask]) + 1e-10)

    harmonic_increase = float(proc_ratio / (orig_ratio + 1e-10) - 1.0)

    # Estimate saturation drive (~1.0 = none, >1.0 = saturated)
    estimated_drive = 1.0 + max(0, harmonic_increase) * 2.0

    return {
        "harmonic_increase": round(harmonic_increase, 3),
        "estimated_drive": round(float(np.clip(estimated_drive, 1.0, 3.0)), 2),
    }


# ---------------------------------------------------------------------------
# Profile Generation
# ---------------------------------------------------------------------------

def generate_profile(
    original_path: str, processed_path: str, n_eq_bands: int = 10,
) -> Dict:
    """Full analysis: generate a DSP profile from original → processed pair."""

    print("Loading audio files...")
    orig_data, orig_sr = sf.read(original_path, dtype="float32")
    proc_data, proc_sr = sf.read(processed_path, dtype="float32")

    if orig_data.ndim == 1:
        orig_data = orig_data.reshape(1, -1)
    elif orig_data.ndim == 2:
        orig_data = orig_data.T

    if proc_data.ndim == 1:
        proc_data = proc_data.reshape(1, -1)
    elif proc_data.ndim == 2:
        proc_data = proc_data.T

    print(f"  Original: {orig_data.shape}, {orig_sr}Hz")
    print(f"  Processed: {proc_data.shape}, {proc_sr}Hz")

    if orig_sr != proc_sr:
        print(f"  WARNING: Sample rates differ ({orig_sr} vs {proc_sr})")

    sr = orig_sr

    # 1. EQ Transfer Function
    print("\n--- Extracting EQ Transfer Function ---")
    freqs, eq_shape_db, overall_gain_db = extract_transfer_function(orig_data, proc_data, sr)

    print(f"  Overall gain (Maximizer): {overall_gain_db:+.1f} dB")
    print("  EQ shape (gain-normalized, dB change):")
    key_freqs = [50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000, 15000]
    for kf in key_freqs:
        idx = np.argmin(np.abs(freqs - kf))
        print(f"    {kf:>6}Hz: {eq_shape_db[idx]:+.1f} dB")

    # 2. Fit EQ Bands (to the shape, not the gain)
    print(f"\n--- Fitting {n_eq_bands} Parametric EQ Bands ---")
    eq_bands = fit_eq_bands(freqs, eq_shape_db, sr, n_bands=n_eq_bands)
    for band in eq_bands:
        print(f"  {band['type']:>10s} @ {band['freq_hz']:>8.0f}Hz: {band['gain_db']:+.1f}dB  Q={band['q']:.1f}")

    # 3. Dynamics
    print("\n--- Analyzing Dynamics ---")
    dynamics = analyze_dynamics(orig_data, proc_data, sr)
    for k, v in dynamics.items():
        print(f"  {k}: {v}")

    # 4. Stereo
    print("\n--- Analyzing Stereo ---")
    stereo = analyze_stereo(orig_data, proc_data, sr)
    for k, v in stereo.items():
        print(f"  {k}: {v}")

    # 5. Exciter
    print("\n--- Analyzing Exciter/Saturation ---")
    exciter = analyze_exciter(orig_data, proc_data, sr)
    for k, v in exciter.items():
        print(f"  {k}: {v}")

    # Build profile
    profile = {
        "source_original": str(Path(original_path).name),
        "source_processed": str(Path(processed_path).name),
        "sample_rate": sr,
        "overall_gain_db": round(overall_gain_db, 1),
        "eq_bands": eq_bands,
        "dynamics": dynamics,
        "stereo": stereo,
        "exciter": exciter,
        "transfer_function": {
            "description": "Gain-normalized 1/3-octave smoothed EQ shape",
            "overall_gain_db": round(overall_gain_db, 1),
            "frequencies_hz": [float(f) for f in freqs[freqs <= 20000][::4]],
            "magnitude_db": [float(m) for m in eq_shape_db[freqs <= 20000][::4]],
        },
    }

    return profile


# ---------------------------------------------------------------------------
# Apply Profile
# ---------------------------------------------------------------------------

def apply_profile(audio_path: str, profile: Dict, output_path: str):
    """Apply a learned DSP profile to an audio file using pedalboard.

    Gain staging order (matches how Ozone works internally):
      1. EQ shape (relative boosts/cuts only)
      2. Light saturation (exciter warmth)
      3. Stereo widening (imager)
      4. Compression (dynamics taming)
      5. Peak-normalize to -1dB headroom
      6. Limiter at -0.3dB (Maximizer — achieves loudness safely)
    """
    print(f"\n--- Applying Profile to {Path(audio_path).name} ---")

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        data = data.T
    print(f"  Input: {data.shape}, {sr}Hz")

    result = data.copy()

    # --- Step 1: EQ shape (relative boosts/cuts, NOT overall gain) ---
    eq_plugins = []
    for band in profile["eq_bands"]:
        if band["type"] == "low_shelf":
            eq_plugins.append(LowShelfFilter(
                cutoff_frequency_hz=band["freq_hz"],
                gain_db=band["gain_db"],
                q=band.get("q", 0.7),
            ))
        elif band["type"] == "high_shelf":
            eq_plugins.append(HighShelfFilter(
                cutoff_frequency_hz=band["freq_hz"],
                gain_db=band["gain_db"],
                q=band.get("q", 0.7),
            ))
        else:  # peak
            eq_plugins.append(PeakFilter(
                cutoff_frequency_hz=band["freq_hz"],
                gain_db=band["gain_db"],
                q=band.get("q", 1.0),
            ))

    if eq_plugins:
        eq_board = Pedalboard(eq_plugins)
        print(f"  Step 1: EQ ({len(eq_plugins)} bands)")
        for p in eq_plugins:
            print(f"    {p}")
        for ch in range(result.shape[0]):
            result[ch] = eq_board.process(result[ch], sample_rate=sr)

    # --- Step 2: Light saturation (exciter) ---
    exc = profile.get("exciter", {})
    drive = exc.get("estimated_drive", 1.0)
    # Cap drive at 1.5 — the detected 2.45 is inflated by EQ changes in
    # the high end, not actual distortion
    drive = min(drive, 1.5)
    if drive > 1.05:
        print(f"  Step 2: Saturation (drive={drive:.2f})")
        result = np.tanh(result * drive) / np.tanh(drive)

    # --- Step 3: Stereo widening (imager) ---
    st = profile.get("stereo", {})
    width_change = st.get("width_change", 0.0)
    if width_change > 0.05 and result.shape[0] >= 2:
        print(f"  Step 3: Stereo widening ({width_change:.1%})")
        mid = (result[0] + result[1]) * 0.5
        side = (result[0] - result[1]) * 0.5
        side *= (1.0 + width_change)
        result[0] = mid + side
        result[1] = mid - side

    # --- Step 4: Compression ---
    dyn = profile.get("dynamics", {})
    if dyn.get("estimated_ratio", 1.0) > 1.2:
        comp = Pedalboard([Compressor(
            threshold_db=dyn.get("estimated_threshold_db", -12),
            ratio=dyn["estimated_ratio"],
            attack_ms=10.0,
            release_ms=100.0,
        )])
        print(f"  Step 4: Compression (ratio={dyn['estimated_ratio']:.1f}, "
              f"threshold={dyn['estimated_threshold_db']:.0f}dB)")
        for ch in range(result.shape[0]):
            result[ch] = comp.process(result[ch], sample_rate=sr)

    # --- Step 5: Loudness push + Limiter (Maximizer) ---
    # Push gain into the limiter to raise average loudness (like Ozone's
    # Maximizer). The limiter catches peaks, the gain raises the floor.
    overall_gain = profile.get("overall_gain_db", 0.0)
    push_db = min(overall_gain * 0.5, 6.0)  # conservative: half the detected gain, max 6dB
    if push_db > 0.5:
        gain_push = Pedalboard([Gain(gain_db=push_db)])
        for ch in range(result.shape[0]):
            result[ch] = gain_push.process(result[ch], sample_rate=sr)
        print(f"  Step 5a: Gain push (+{push_db:.1f}dB)")

    limiter = Pedalboard([Limiter(threshold_db=-0.5, release_ms=50.0)])
    for ch in range(result.shape[0]):
        result[ch] = limiter.process(result[ch], sample_rate=sr)
    print(f"  Step 5b: Limiter (-0.5dBFS)")

    # --- Step 6: Final peak-normalize to -0.1dBFS (max loudness, no clipping) ---
    peak = np.max(np.abs(result))
    if peak > 0.001:
        target_peak = 10 ** (-0.1 / 20)  # -0.1 dBFS
        result *= target_peak / peak
        print(f"  Step 6: Peak-normalized to -0.1dBFS (was {20*np.log10(peak):+.1f}dBFS)")

    print(f"  Saving to: {output_path}")
    sf.write(output_path, result.T, sr, subtype="FLOAT")
    print("  Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Learn DSP profile from reference audio pair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("original", help="Path to original audio file")
    parser.add_argument("processed", help="Path to Ozone-processed audio file")
    parser.add_argument("--n-bands", type=int, default=10,
                        help="Number of EQ bands to fit (default: 10)")
    parser.add_argument("--output-profile", type=str, default=None,
                        help="Output JSON profile path (default: auto)")
    parser.add_argument("--apply", type=str, default=None,
                        help="Apply learned profile to this audio file")
    parser.add_argument("--apply-output", type=str, default=None,
                        help="Output path for --apply (default: auto)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.original).exists():
        print(f"Error: {args.original} not found")
        sys.exit(1)
    if not Path(args.processed).exists():
        print(f"Error: {args.processed} not found")
        sys.exit(1)

    # Generate profile
    profile = generate_profile(args.original, args.processed, args.n_bands)

    # Save profile
    profile_path = args.output_profile or str(
        Path(args.processed).parent / f"{Path(args.processed).stem}_profile.json"
    )
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\n✓ Profile saved to: {profile_path}")

    # Apply if requested
    if args.apply:
        if not Path(args.apply).exists():
            print(f"Error: {args.apply} not found")
            sys.exit(1)
        apply_output = args.apply_output or str(
            Path(args.apply).parent / f"{Path(args.apply).stem}_profiled.wav"
        )
        apply_profile(args.apply, profile, apply_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
