"""
Auto-Mastering Engine
=====================
Applies a mastering profile (EQ, compression, saturation, stereo
widening, limiting) to generated audio.  Supports multiple presets,
per-parameter overrides, and custom preset save/delete.

Usage::

    from acestep.core.audio.mastering import MasteringEngine

    engine = MasteringEngine()                     # loads default preset
    engine = MasteringEngine(preset_name="ozone1") # loads specific preset

    mastered = engine.master(audio_np, sample_rate)
    mastered = engine.master(audio_np, sample_rate, params_override={...})

    # Preset management
    presets = MasteringEngine.list_presets()
    MasteringEngine.save_preset("my_preset", params_dict)
    MasteringEngine.delete_preset("my_preset")
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Presets live alongside this module
_PRESETS_DIR = Path(__file__).parent / "presets"
# Legacy profile for backward compat
_LEGACY_PROFILE = Path(__file__).parent / "mastering_profile.json"

# Built-in preset IDs that cannot be deleted
_BUILTIN_PRESETS = {"preset_1", "preset_2"}


class MasteringEngine:
    """Profile-based mastering processor using pedalboard.

    Gain staging order (matches professional mastering signal flow):
      1. EQ shape (relative boosts/cuts only, no overall gain)
      2. Light saturation (exciter warmth)
      3. Stereo widening (imager)
      4. Compression (dynamics taming)
      5. Loudness push via gain + limiter (Maximizer equivalent)
      6. Peak-normalize to -0.1 dBFS (max loudness, no clipping)
    """

    def __init__(self, profile_path: Optional[str] = None, preset_name: Optional[str] = None):
        """Load a mastering profile.

        Args:
            profile_path: Path to a JSON profile file. If None, uses preset_name.
            preset_name: Name of a preset to load (e.g. "default", "ozone1").
                         If both profile_path and preset_name are None, loads "default".
        """
        if profile_path:
            path = Path(profile_path)
        elif preset_name:
            path = _PRESETS_DIR / f"{preset_name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Preset not found: {preset_name}")
        else:
            # Default: try presets dir first, fall back to legacy
            path = _PRESETS_DIR / "default.json"
            if not path.exists():
                path = _LEGACY_PROFILE

        if not path.exists():
            raise FileNotFoundError(f"Mastering profile not found: {path}")

        with open(path, "r") as f:
            self.profile: Dict = json.load(f)

        self._preset_id = preset_name or path.stem
        logger.info(f"[Mastering] Loaded preset '{self._preset_id}' from {path.name}")

    def get_params(self) -> Dict:
        """Return the current profile as a dict (for frontend preset loading)."""
        return self.profile.copy()

    @staticmethod
    def list_presets() -> List[Dict]:
        """List all available presets (built-in + custom).

        Returns list of dicts: [{id, name, description, builtin, params}]
        """
        presets = []
        if not _PRESETS_DIR.exists():
            return presets

        for path in sorted(_PRESETS_DIR.glob("*.json")):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                preset_id = path.stem
                presets.append({
                    "id": preset_id,
                    "name": data.get("name", preset_id.replace("_", " ").title()),
                    "description": data.get("description", ""),
                    "builtin": preset_id in _BUILTIN_PRESETS,
                    "params": data,
                })
            except Exception as e:
                logger.warning(f"[Mastering] Failed to load preset {path.name}: {e}")

        return presets

    @staticmethod
    def save_preset(name: str, params: Dict) -> str:
        """Save a custom preset.

        Args:
            name: Human-readable preset name.
            params: Full mastering parameters dict.

        Returns:
            The preset ID (filename stem).
        """
        _PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate safe filename from name
        preset_id = re.sub(r'[^\w\-]', '_', name.lower().strip())
        if not preset_id:
            preset_id = "custom"

        # Don't overwrite built-ins
        if preset_id in _BUILTIN_PRESETS:
            preset_id = f"custom_{preset_id}"

        # Ensure name/description are in the params
        params = params.copy()
        params["name"] = name
        if "description" not in params:
            params["description"] = f"Custom preset: {name}"

        path = _PRESETS_DIR / f"{preset_id}.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

        logger.info(f"[Mastering] Saved preset '{name}' → {path.name}")
        return preset_id

    @staticmethod
    def delete_preset(preset_id: str) -> bool:
        """Delete a custom preset. Built-in presets cannot be deleted.

        Returns True if deleted, False if not found or protected.
        """
        if preset_id in _BUILTIN_PRESETS:
            logger.warning(f"[Mastering] Cannot delete built-in preset: {preset_id}")
            return False

        path = _PRESETS_DIR / f"{preset_id}.json"
        if not path.exists():
            return False

        path.unlink()
        logger.info(f"[Mastering] Deleted preset: {preset_id}")
        return True

    def master(self, audio: np.ndarray, sample_rate: int,
               params_override: Optional[Dict] = None) -> np.ndarray:
        """Apply the mastering profile to audio.

        Args:
            audio: Audio array, shape [channels, samples] (float32).
            sample_rate: Sample rate in Hz.
            params_override: Optional dict of parameter overrides. When provided,
                             these values are used instead of the stored profile.
                             Keys match the profile JSON structure.

        Returns:
            Mastered audio, same shape as input.
        """
        from pedalboard import (
            Pedalboard, Compressor, LowShelfFilter, HighShelfFilter,
            PeakFilter, Gain, Limiter,
        )

        # Merge override with profile (override wins)
        if params_override:
            profile = {**self.profile, **params_override}
            # Deep-merge nested dicts
            for key in ("dynamics", "stereo", "exciter"):
                if key in params_override and key in self.profile:
                    profile[key] = {**self.profile[key], **params_override[key]}
        else:
            profile = self.profile

        result = audio.copy()

        # --- Step 1: EQ shape ---
        eq_plugins = []
        for band in profile.get("eq_bands", []):
            gain = band["gain_db"]
            if abs(gain) < 0.05:
                continue  # skip negligible bands
            if band["type"] == "low_shelf":
                eq_plugins.append(LowShelfFilter(
                    cutoff_frequency_hz=band["freq_hz"],
                    gain_db=gain,
                    q=band.get("q", 0.7),
                ))
            elif band["type"] == "high_shelf":
                eq_plugins.append(HighShelfFilter(
                    cutoff_frequency_hz=band["freq_hz"],
                    gain_db=gain,
                    q=band.get("q", 0.7),
                ))
            else:  # peak
                eq_plugins.append(PeakFilter(
                    cutoff_frequency_hz=band["freq_hz"],
                    gain_db=gain,
                    q=band.get("q", 1.0),
                ))

        if eq_plugins:
            eq_board = Pedalboard(eq_plugins)
            for ch in range(result.shape[0]):
                result[ch] = eq_board.process(result[ch], sample_rate=sample_rate)

        # --- Step 2: Light saturation (exciter) ---
        exc = profile.get("exciter", {})
        drive = min(exc.get("estimated_drive", 1.0), 3.0)
        if drive > 1.05:
            result = np.tanh(result * drive) / np.tanh(drive)

        # --- Step 3: Stereo widening ---
        st = profile.get("stereo", {})
        width_change = st.get("width_change", 0.0)
        if width_change > 0.01 and result.shape[0] >= 2:
            mid = (result[0] + result[1]) * 0.5
            side = (result[0] - result[1]) * 0.5
            side *= (1.0 + width_change)
            result[0] = mid + side
            result[1] = mid - side

        # --- Step 4: Compression ---
        dyn = profile.get("dynamics", {})
        ratio = dyn.get("estimated_ratio", 1.0)
        if ratio > 1.1:
            comp = Pedalboard([Compressor(
                threshold_db=dyn.get("estimated_threshold_db", -12),
                ratio=ratio,
                attack_ms=10.0,
                release_ms=100.0,
            )])
            for ch in range(result.shape[0]):
                result[ch] = comp.process(result[ch], sample_rate=sample_rate)

        # --- Step 5: Loudness push + Limiter (Maximizer) ---
        overall_gain = profile.get("overall_gain_db", 0.0)
        push_db = min(overall_gain * 0.5, 6.0)
        ceiling_db = profile.get("limiter_ceiling_db", -0.5)
        if push_db > 0.5:
            gain_push = Pedalboard([Gain(gain_db=push_db)])
            for ch in range(result.shape[0]):
                result[ch] = gain_push.process(result[ch], sample_rate=sample_rate)

        limiter = Pedalboard([Limiter(threshold_db=ceiling_db, release_ms=50.0)])
        for ch in range(result.shape[0]):
            result[ch] = limiter.process(result[ch], sample_rate=sample_rate)

        # --- Step 6: Final peak-normalize to -0.1 dBFS ---
        peak = np.max(np.abs(result))
        if peak > 0.001:
            target_peak = 10 ** (-0.1 / 20)  # -0.1 dBFS
            result *= target_peak / peak

        return result
