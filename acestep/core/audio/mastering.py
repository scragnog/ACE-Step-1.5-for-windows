"""
Auto-Mastering Engine
=====================
Applies a learned mastering profile (EQ, compression, saturation, stereo
widening, limiting) to generated audio. The default profile was learned
from an iZotope Ozone 11 "High Detail Wide" processing chain by comparing
original and processed reference tracks.

Usage in the generation pipeline::

    from acestep.core.audio.mastering import MasteringEngine

    engine = MasteringEngine()  # loads bundled profile
    mastered_np = engine.master(audio_np, sample_rate)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bundled default profile (learned from Ozone 11 "High Detail Wide")
_DEFAULT_PROFILE = Path(__file__).parent / "mastering_profile.json"


class MasteringEngine:
    """Profile-based mastering processor using pedalboard.

    Gain staging order (matches how Ozone works internally):
      1. EQ shape (relative boosts/cuts only, no overall gain)
      2. Light saturation (exciter warmth)
      3. Stereo widening (imager)
      4. Compression (dynamics taming)
      5. Loudness push via gain + limiter (Maximizer equivalent)
      6. Peak-normalize to -0.1 dBFS (max loudness, no clipping)
    """

    def __init__(self, profile_path: Optional[str] = None):
        """Load a mastering profile.

        Args:
            profile_path: Path to a JSON profile file. If None, uses the
                          bundled default profile.
        """
        path = Path(profile_path) if profile_path else _DEFAULT_PROFILE
        if not path.exists():
            raise FileNotFoundError(f"Mastering profile not found: {path}")

        with open(path, "r") as f:
            self.profile: Dict = json.load(f)

        self._board_cache = None
        logger.info(f"[Mastering] Loaded profile from {path.name}")

    def master(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply the mastering profile to audio.

        Args:
            audio: Audio array, shape [channels, samples] (float32).
            sample_rate: Sample rate in Hz.

        Returns:
            Mastered audio, same shape as input.
        """
        from pedalboard import (
            Pedalboard, Compressor, LowShelfFilter, HighShelfFilter,
            PeakFilter, Gain, Limiter,
        )

        result = audio.copy()
        profile = self.profile

        # --- Step 1: EQ shape ---
        eq_plugins = []
        for band in profile.get("eq_bands", []):
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
            for ch in range(result.shape[0]):
                result[ch] = eq_board.process(result[ch], sample_rate=sample_rate)

        # --- Step 2: Light saturation (exciter) ---
        exc = profile.get("exciter", {})
        drive = min(exc.get("estimated_drive", 1.0), 1.5)
        if drive > 1.05:
            result = np.tanh(result * drive) / np.tanh(drive)

        # --- Step 3: Stereo widening ---
        st = profile.get("stereo", {})
        width_change = st.get("width_change", 0.0)
        if width_change > 0.05 and result.shape[0] >= 2:
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
            for ch in range(result.shape[0]):
                result[ch] = comp.process(result[ch], sample_rate=sample_rate)

        # --- Step 5: Loudness push + Limiter (Maximizer) ---
        overall_gain = profile.get("overall_gain_db", 0.0)
        push_db = min(overall_gain * 0.5, 6.0)
        if push_db > 0.5:
            gain_push = Pedalboard([Gain(gain_db=push_db)])
            for ch in range(result.shape[0]):
                result[ch] = gain_push.process(result[ch], sample_rate=sample_rate)

        limiter = Pedalboard([Limiter(threshold_db=-0.5, release_ms=50.0)])
        for ch in range(result.shape[0]):
            result[ch] = limiter.process(result[ch], sample_rate=sample_rate)

        # --- Step 6: Final peak-normalize to -0.1 dBFS ---
        peak = np.max(np.abs(result))
        if peak > 0.001:
            target_peak = 10 ** (-0.1 / 20)  # -0.1 dBFS
            result *= target_peak / peak

        return result
