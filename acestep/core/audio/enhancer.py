"""
Audio Quality Enhancement Engine
================================
Ported from ComfyUI-Audio_Quality_Enhancer by ShmuelRonen.
Provides multi-band EQ, compression, reverb/echo, stereo widening,
and optional Demucs stem-separation for targeted per-stem enhancement.

Dependencies:
  - Required: numpy, scipy, soundfile
  - Optional: pedalboard (better EQ/compression), demucs (stem separation)
"""

import os
import uuid
import numpy as np
import soundfile as sf
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any
from scipy import signal


# ---------------------------------------------------------------------------
# BFloat16 workaround (same fix as stem_service.py)
# ACE-Step sets torch default dtype to bfloat16 for GPU inference.
# audio_separator's Demucs creates internal tensors that inherit that dtype,
# causing MKL FFT crash on Windows.
# ---------------------------------------------------------------------------

@contextmanager
def _float32_default_dtype():
    """Temporarily force torch default dtype to float32."""
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

# Optional: pedalboard for higher-quality DSP
try:
    from pedalboard import (
        Pedalboard, Compressor, LowShelfFilter, HighShelfFilter,
        Gain, PeakFilter, Limiter
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# Optional: Demucs for stem separation
# First try audio_separator (which bundles Demucs internally — this is what
# the stem separation feature uses).  Fall back to standalone demucs package.
DEMUCS_AVAILABLE = False
_DEMUCS_BACKEND = None          # "audio_separator" or "standalone"
try:
    from audio_separator.separator import Separator  # noqa: F401
    DEMUCS_AVAILABLE = True
    _DEMUCS_BACKEND = "audio_separator"
except ImportError:
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch
        DEMUCS_AVAILABLE = True
        _DEMUCS_BACKEND = "standalone"
    except ImportError:
        pass

# Optional: librosa for resampling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


# ---- Presets ----

PRESETS = {
    "radio_ready": {
        "label": "Radio Ready",
        "clarity": 0.6, "warmth": 0.3, "air": 0.5, "dynamics": 0.6,
        "reverb_amount": 0.0, "reverb_room_size": 0.4, "reverb_damping": 0.5,
        "echo_delay": 0.0, "echo_decay": 0.0,
        "stereo_width": 0.2,
        "vocals_enhance": 0.6, "drums_enhance": 0.5, "bass_enhance": 0.4, "other_enhance": 0.4,
    },
    "warm_and_rich": {
        "label": "Warm & Rich",
        "clarity": 0.3, "warmth": 0.6, "air": 0.2, "dynamics": 0.3,
        "reverb_amount": 0.15, "reverb_room_size": 0.5, "reverb_damping": 0.6,
        "echo_delay": 0.0, "echo_decay": 0.0,
        "stereo_width": 0.1,
        "vocals_enhance": 0.4, "drums_enhance": 0.3, "bass_enhance": 0.6, "other_enhance": 0.5,
    },
    "bright_and_clear": {
        "label": "Bright & Clear",
        "clarity": 0.7, "warmth": 0.1, "air": 0.7, "dynamics": 0.4,
        "reverb_amount": 0.0, "reverb_room_size": 0.3, "reverb_damping": 0.4,
        "echo_delay": 0.0, "echo_decay": 0.0,
        "stereo_width": 0.15,
        "vocals_enhance": 0.7, "drums_enhance": 0.4, "bass_enhance": 0.2, "other_enhance": 0.5,
    },
    "club_master": {
        "label": "Club Master",
        "clarity": 0.4, "warmth": 0.5, "air": 0.4, "dynamics": 0.7,
        "reverb_amount": 0.0, "reverb_room_size": 0.3, "reverb_damping": 0.3,
        "echo_delay": 0.0, "echo_decay": 0.0,
        "stereo_width": 0.3,
        "vocals_enhance": 0.3, "drums_enhance": 0.7, "bass_enhance": 0.7, "other_enhance": 0.3,
    },
    "lo_fi_chill": {
        "label": "Lo-Fi Chill",
        "clarity": 0.2, "warmth": 0.7, "air": 0.1, "dynamics": 0.2,
        "reverb_amount": 0.3, "reverb_room_size": 0.6, "reverb_damping": 0.7,
        "echo_delay": 0.25, "echo_decay": 0.3,
        "stereo_width": 0.1,
        "vocals_enhance": 0.3, "drums_enhance": 0.2, "bass_enhance": 0.5, "other_enhance": 0.4,
    },
    "cinematic": {
        "label": "Cinematic",
        "clarity": 0.5, "warmth": 0.4, "air": 0.6, "dynamics": 0.5,
        "reverb_amount": 0.4, "reverb_room_size": 0.8, "reverb_damping": 0.5,
        "echo_delay": 0.15, "echo_decay": 0.2,
        "stereo_width": 0.4,
        "vocals_enhance": 0.5, "drums_enhance": 0.4, "bass_enhance": 0.5, "other_enhance": 0.6,
    },
}


# ---- Core DSP Functions ----

def _apply_eq_pedalboard(audio: np.ndarray, sample_rate: int,
                         warmth: float, clarity: float, air: float,
                         dynamics: float) -> np.ndarray:
    """Apply EQ and compression using pedalboard (high quality)."""
    plugins = []
    if warmth > 0:
        plugins.append(LowShelfFilter(cutoff_frequency_hz=100, gain_db=warmth * 4, q=0.7))
    if clarity > 0:
        plugins.append(PeakFilter(cutoff_frequency_hz=2500, gain_db=clarity * 5, q=1.0))
    if air > 0:
        plugins.append(HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air * 5, q=0.7))
    if dynamics > 0:
        plugins.append(Compressor(
            threshold_db=-20, ratio=1.5 + (dynamics * 1.5),
            attack_ms=5.0, release_ms=50.0
        ))
        plugins.append(Gain(gain_db=dynamics * 3))

    if not plugins:
        return audio

    board = Pedalboard(plugins)
    result = audio.copy()
    for ch in range(result.shape[0]):
        result[ch] = board.process(result[ch], sample_rate=sample_rate)
    return result


def _apply_eq_scipy(audio: np.ndarray, sample_rate: int,
                    warmth: float, clarity: float, air: float,
                    dynamics: float) -> np.ndarray:
    """Apply EQ using scipy butterworth filters (fallback)."""
    result = audio.copy()
    nyquist = sample_rate / 2

    # Warmth: boost low frequencies
    if warmth > 0:
        low_cutoff = min(200 / nyquist, 0.99)
        low_sos = signal.butter(2, low_cutoff, btype='lowpass', output='sos')
        low_band = signal.sosfilt(low_sos, result)
        result = result + low_band * warmth * 0.3

    # Clarity: boost presence (2-5kHz)
    if clarity > 0:
        presence_low = min(2000 / nyquist, 0.99)
        presence_high = min(5000 / nyquist, 0.99)
        if presence_low < presence_high:
            presence_sos = signal.butter(2, [presence_low, presence_high], btype='bandpass', output='sos')
            presence = signal.sosfilt(presence_sos, result)
            result = result + presence * clarity * 0.5

    # Air: boost high frequencies
    if air > 0:
        high_cutoff = min(8000 / nyquist, 0.99)
        high_sos = signal.butter(2, high_cutoff, btype='highpass', output='sos')
        high_band = signal.sosfilt(high_sos, result)
        result = result + high_band * air * 0.3

    # Dynamics: simple transient enhancement
    if dynamics > 0:
        for ch in range(result.shape[0]):
            env = np.abs(result[ch])
            win_size = max(int(0.01 * sample_rate), 1)
            env = np.convolve(env, np.ones(win_size) / win_size, mode='same')
            env_diff = np.zeros_like(env)
            env_diff[1:] = env[1:] - env[:-1]
            transient_mask = env_diff > 0.01
            result[ch, transient_mask] *= (1.0 + dynamics * 0.5)

    return result


def apply_eq(audio: np.ndarray, sample_rate: int,
             warmth: float, clarity: float, air: float,
             dynamics: float) -> np.ndarray:
    """Apply EQ — uses pedalboard if available, scipy fallback otherwise."""
    if PEDALBOARD_AVAILABLE:
        return _apply_eq_pedalboard(audio, sample_rate, warmth, clarity, air, dynamics)
    return _apply_eq_scipy(audio, sample_rate, warmth, clarity, air, dynamics)


# ---- Reverb (impulse response convolution, no SoX) ----

def generate_impulse_response(sample_rate: int, room_size: float = 0.5,
                              damping: float = 0.5, duration: float = 2.0) -> np.ndarray:
    """Generate a synthetic reverb impulse response."""
    num_samples = int(sample_rate * duration * room_size)
    if num_samples < 1:
        return np.array([1.0])

    # Exponential decay envelope
    decay_rate = 3.0 + damping * 7.0  # Higher damping = faster decay
    t = np.linspace(0, duration * room_size, num_samples)
    envelope = np.exp(-decay_rate * t)

    # Random noise shaped by envelope (simulates diffuse reflections)
    ir = np.random.randn(num_samples) * envelope

    # Add some early reflections
    for delay_ms in [15, 25, 40, 65]:
        delay_samples = int(delay_ms / 1000 * sample_rate)
        if delay_samples < num_samples:
            reflection_strength = 0.3 * np.exp(-delay_ms / 50)
            ir[delay_samples] += reflection_strength

    # Normalize
    max_val = np.max(np.abs(ir))
    if max_val > 0:
        ir = ir / max_val

    return ir


def apply_reverb(audio: np.ndarray, sample_rate: int,
                 amount: float = 0.3, room_size: float = 0.5,
                 damping: float = 0.5) -> np.ndarray:
    """Apply reverb using convolution with synthetic impulse response."""
    if amount <= 0:
        return audio

    ir = generate_impulse_response(sample_rate, room_size, damping)
    result = audio.copy()

    for ch in range(result.shape[0]):
        # Convolve with impulse response
        wet = signal.fftconvolve(result[ch], ir, mode='full')[:result.shape[1]]
        # Mix dry/wet
        result[ch] = result[ch] * (1 - amount) + wet * amount

    return result


# ---- Echo (delay line) ----

def apply_echo(audio: np.ndarray, sample_rate: int,
               delay: float = 0.3, decay: float = 0.4) -> np.ndarray:
    """Apply echo effect using delay line with feedback."""
    if delay <= 0 or decay <= 0:
        return audio

    delay_samples = int(delay * sample_rate)
    if delay_samples < 1:
        return audio

    result = audio.copy()
    num_repeats = 4  # Number of echo repetitions

    for ch in range(result.shape[0]):
        for i in range(1, num_repeats + 1):
            offset = delay_samples * i
            strength = decay ** i
            if offset >= result.shape[1] or strength < 0.01:
                break
            end = min(result.shape[1], result.shape[1] - offset + offset)
            result[ch, offset:] += result[ch, :result.shape[1] - offset] * strength

    return result


# ---- Stereo Widening (mid/side processing) ----

def apply_stereo_widening(audio: np.ndarray, sample_rate: int,
                          amount: float = 0.3) -> np.ndarray:
    """Apply Dolby-like stereo widening using mid/side processing."""
    if amount <= 0 or audio.shape[0] < 2:
        return audio

    # Mid/side decomposition
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5

    nyquist = sample_rate / 2

    # Keep bass centered (below 150Hz)
    low_cut = min(150 / nyquist, 0.99)
    side_sos = signal.butter(2, low_cut, btype='highpass', output='sos')
    filtered_side = signal.sosfilt(side_sos, side)

    # Boost upper mids/highs in side channel
    high_boost = min(2000 / nyquist, 0.99)
    high_sos = signal.butter(2, high_boost, btype='highpass', output='sos')
    high_side = signal.sosfilt(high_sos, filtered_side)

    # Mild saturation for presence
    high_side = np.tanh(high_side * (1.0 + amount * 1.5)) / (1.0 + amount * 0.5)

    enhanced_side = filtered_side + high_side * amount * 1.5

    # Haas effect (small delay for increased width)
    if amount > 0.2:
        delay_samples = int(sample_rate * 0.015 * amount)
        if delay_samples > 0 and delay_samples < len(enhanced_side):
            enhanced_side = np.concatenate([
                np.zeros(delay_samples),
                enhanced_side[:-delay_samples]
            ])

    # Bass management
    bass_cutoff = min(150 / nyquist, 0.99)
    bass_sos = signal.butter(2, bass_cutoff, btype='lowpass', output='sos')
    bass = signal.sosfilt(bass_sos, mid)
    enhanced_bass = np.tanh(bass * (1.0 + amount * 0.8)) / (1.0 + amount * 0.2)

    # Presence enhancement in mid channel
    presence_low = min(1000 / nyquist, 0.99)
    presence_high = min(5000 / nyquist, 0.99)
    if presence_low < presence_high:
        presence_sos = signal.butter(2, [presence_low, presence_high], btype='bandpass', output='sos')
        presence = signal.sosfilt(presence_sos, mid)
        enhanced_mid = mid + presence * amount * 0.4
    else:
        enhanced_mid = mid

    # Air band (>10kHz)
    if sample_rate > 30000:
        air_cutoff = min(10000 / nyquist, 0.99)
        air_sos = signal.butter(2, air_cutoff, btype='highpass', output='sos')
        air_band = signal.sosfilt(air_sos, mid)
        enhanced_mid = enhanced_mid + air_band * amount * 0.5

    # Recombine
    side_level = 1.0 + amount * 1.0
    left = enhanced_mid + enhanced_side * side_level + enhanced_bass
    right = enhanced_mid - enhanced_side * side_level + enhanced_bass

    # Final saturation for cohesion
    if amount > 0.5:
        saturation = 1.0 + (amount - 0.5) * 0.6
        left = np.tanh(left * saturation) / saturation
        right = np.tanh(right * saturation) / saturation

    result = np.vstack([left, right])

    # Normalize
    max_val = np.max(np.abs(result))
    if max_val > 0.98:
        result = result * (0.98 / max_val)

    return result


# ---- Per-Stem Enhancement (for Demucs mode) ----

def _enhance_vocals(vocals: np.ndarray, sample_rate: int,
                    level: float, clarity: float, air: float) -> np.ndarray:
    """Enhance vocals with focus on clarity and presence."""
    if level <= 0:
        return vocals

    result = vocals.copy()

    if PEDALBOARD_AVAILABLE:
        plugins = [
            PeakFilter(cutoff_frequency_hz=3500, gain_db=clarity * 6, q=1.0),
            PeakFilter(cutoff_frequency_hz=7500, gain_db=-clarity * 2, q=2.0),  # De-ess
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air * 4, q=0.7),
        ]
        board = Pedalboard(plugins)
        for ch in range(result.shape[0]):
            result[ch] = board.process(result[ch], sample_rate=sample_rate)
    else:
        nyquist = sample_rate / 2
        presence_low = min(3000 / nyquist, 0.99)
        presence_high = min(4000 / nyquist, 0.99)
        if presence_low < presence_high:
            presence_sos = signal.butter(2, [presence_low, presence_high], btype='bandpass', output='sos')
            presence = signal.sosfilt(presence_sos, result)
            result = result + presence * clarity * 0.5

    return result


def _enhance_drums(drums: np.ndarray, sample_rate: int,
                   level: float, dynamics: float, air: float) -> np.ndarray:
    """Enhance drums with transient boost and high-end air."""
    if level <= 0:
        return drums

    result = drums.copy()

    # Transient enhancement
    for ch in range(result.shape[0]):
        env = np.abs(result[ch])
        win_size = max(int(0.01 * sample_rate), 1)
        env = np.convolve(env, np.ones(win_size) / win_size, mode='same')
        env_diff = np.zeros_like(env)
        env_diff[1:] = env[1:] - env[:-1]
        transient_mask = env_diff > 0.01
        result[ch, transient_mask] *= (1.0 + level * 0.7)

    # Air for cymbals
    if air > 0 and PEDALBOARD_AVAILABLE:
        board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=air * 6, q=0.7)
        ])
        for ch in range(result.shape[0]):
            result[ch] = board.process(result[ch], sample_rate=sample_rate)

    return result


def _enhance_bass(bass: np.ndarray, sample_rate: int,
                  level: float, warmth: float) -> np.ndarray:
    """Enhance bass with warmth and harmonic definition."""
    if level <= 0:
        return bass

    result = bass.copy()

    if PEDALBOARD_AVAILABLE and warmth > 0:
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=100, gain_db=warmth * 4, q=0.7),
            PeakFilter(cutoff_frequency_hz=250, gain_db=level * 3, q=1.0),
        ])
        for ch in range(result.shape[0]):
            result[ch] = board.process(result[ch], sample_rate=sample_rate)
    else:
        # Simple saturation for definition
        drive = 1.0 + level * 2.0
        result = np.tanh(result * drive) / drive

    return result


def _enhance_other(other: np.ndarray, sample_rate: int,
                   level: float, clarity: float, warmth: float, air: float) -> np.ndarray:
    """Enhance other instruments with balanced EQ."""
    if level <= 0:
        return other

    result = other.copy()

    if PEDALBOARD_AVAILABLE:
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=warmth * 3, q=0.7),
            PeakFilter(cutoff_frequency_hz=2000, gain_db=clarity * 3, q=1.0),
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=air * 4, q=0.7),
        ])
        for ch in range(result.shape[0]):
            result[ch] = board.process(result[ch], sample_rate=sample_rate)

    return result


# ---- Main Enhancement Pipeline ----

class AudioEnhancer:
    """Audio enhancement engine with simple and Demucs modes."""

    def __init__(self):
        self._demucs_model = None
        self._demucs_model_name = None
        # Stem cache: { audio_path_hash: { "stems": Dict[str, np.ndarray], "sample_rate": int } }
        self._stem_cache: Dict[str, Dict[str, Any]] = {}
        self._stem_cache_max = 3  # Keep at most N cached separations

    def _load_demucs(self, model_name: str = "BS-Roformer-SW.ckpt", device: str = "cuda"):
        """Load Demucs model on demand.

        Supports two backends:
          - "audio_separator": uses audio_separator.Separator (bundled Demucs)
          - "standalone": uses demucs.pretrained directly
        """
        if not DEMUCS_AVAILABLE:
            return None
        try:
            if _DEMUCS_BACKEND == "audio_separator":
                if self._demucs_model is None or self._demucs_model_name != model_name:
                    logger.info(f"Loading audio_separator with Demucs model: {model_name}")
                    from audio_separator.separator import Separator
                    with _float32_default_dtype():
                        self._demucs_model = Separator()
                        self._demucs_model.load_model(model_filename=model_name)
                    self._demucs_model_name = model_name
                return self._demucs_model
            else:
                # Standalone demucs
                if self._demucs_model is None or self._demucs_model_name != model_name:
                    logger.info(f"Loading Demucs model: {model_name}")
                    self._demucs_model = get_model(model_name)
                    self._demucs_model_name = model_name
                    self._demucs_model.to(device)
                return self._demucs_model
        except Exception as e:
            logger.error(f"Failed to load Demucs: {e}")
            return None

    def clear_stem_cache(self):
        """Clear cached stem separations."""
        self._stem_cache.clear()
        logger.info("Stem cache cleared")

    def get_available_info(self) -> Dict[str, Any]:
        """Get availability info for dependencies."""
        return {
            "available": True,  # scipy is always available (in requirements)
            "pedalboard": PEDALBOARD_AVAILABLE,
            "demucs": DEMUCS_AVAILABLE,
            "presets": {k: v["label"] for k, v in PRESETS.items()},
        }

    def enhance(self, audio_path: str, output_dir: str,
                params: Dict[str, Any],
                progress_callback=None) -> str:
        """
        Main enhancement entry point.

        Args:
            audio_path: Path to source audio file
            output_dir: Directory to write enhanced file
            params: Enhancement parameters dict
            progress_callback: Optional callable(percent: float, message: str)

        Returns:
            Path to enhanced audio file
        """
        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        report(0.0, "Loading audio…")

        # Load audio
        audio, sample_rate = sf.read(audio_path, dtype='float32')

        # Ensure [channels, samples] shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2:
            audio = audio.T  # soundfile returns [samples, channels] → [channels, samples]

        logger.info(f"Audio loaded: {audio.shape[0]} channels, {audio.shape[1]} samples, {sample_rate}Hz")

        # Extract parameters with defaults
        enhancement_level = params.get("enhancement_level", 0.5)
        use_stems = params.get("use_stem_separation", False)
        clarity = params.get("clarity", 0.4) * enhancement_level
        warmth = params.get("warmth", 0.3) * enhancement_level
        air = params.get("air", 0.3) * enhancement_level
        dynamics = params.get("dynamics", 0.3) * enhancement_level
        reverb_amount = params.get("reverb_amount", 0.0) * enhancement_level
        reverb_room_size = params.get("reverb_room_size", 0.5)
        reverb_damping = params.get("reverb_damping", 0.5)
        echo_delay = params.get("echo_delay", 0.0)
        echo_decay = params.get("echo_decay", 0.0) * enhancement_level
        stereo_width = params.get("stereo_width", 0.0) * enhancement_level

        if use_stems and DEMUCS_AVAILABLE:
            report(0.05, "Separating stems with Demucs…")
            enhanced = self._process_with_demucs(
                audio, sample_rate, params, progress_callback=report
            )
        else:
            report(0.1, "Applying EQ and dynamics…")
            enhanced = apply_eq(audio, sample_rate, warmth, clarity, air, dynamics)

        report(0.7, "Applying effects…")

        # Reverb
        if reverb_amount > 0:
            enhanced = apply_reverb(enhanced, sample_rate, reverb_amount,
                                    reverb_room_size, reverb_damping)

        # Echo
        if echo_delay > 0 and echo_decay > 0:
            enhanced = apply_echo(enhanced, sample_rate, echo_delay, echo_decay)

        # Stereo widening
        if stereo_width > 0:
            enhanced = apply_stereo_widening(enhanced, sample_rate, stereo_width)

        report(0.85, "Applying limiter…")

        # Final limiter
        if PEDALBOARD_AVAILABLE:
            for ch in range(enhanced.shape[0]):
                board = Pedalboard([Limiter(threshold_db=-0.5, release_ms=50.0)])
                max_val = np.max(np.abs(enhanced[ch]))
                if max_val > 1.0:
                    enhanced[ch] = enhanced[ch] / max_val
                enhanced[ch] = board.process(enhanced[ch], sample_rate=sample_rate)
        else:
            max_val = np.max(np.abs(enhanced))
            if max_val > 0.98:
                enhanced = enhanced * (0.98 / max_val)

        report(0.9, "Saving enhanced audio…")

        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"enhanced_{uuid.uuid4().hex[:8]}.wav")
        sf.write(output_path, enhanced.T, sample_rate)  # [channels, samples] → [samples, channels]

        report(1.0, "Done!")
        logger.info(f"Enhanced audio saved to: {output_path}")
        return output_path

    def _process_with_demucs(self, audio: np.ndarray, sample_rate: int,
                             params: Dict[str, Any],
                             progress_callback=None) -> np.ndarray:
        """Process using Demucs stem separation for targeted enhancement.

        Supports two backends:
          - audio_separator: separates to temp WAV files, reads them back
          - standalone demucs: uses tensors directly
        """
        import hashlib

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        enhancement_level = params.get("enhancement_level", 0.5)
        device = params.get("device", "cuda")
        model_name = params.get("demucs_model", "BS-Roformer-SW.ckpt")

        # Stem cache: key by audio content hash so same file reuses stems
        cache_key = hashlib.md5(audio.tobytes()[:1_000_000]).hexdigest()  # hash first ~1MB for speed

        try:
            # Check cache first
            if cache_key in self._stem_cache:
                logger.info(f"Using cached stems for key {cache_key[:8]}…")
                report(0.35, "Using cached stems (skipping separation)…")
                stems = self._stem_cache[cache_key]["stems"]
            else:
                model = self._load_demucs(model_name, device)
                if model is None:
                    raise RuntimeError("Failed to load Demucs model")

                report(0.1, f"Running Demucs ({model_name})…")

                if _DEMUCS_BACKEND == "audio_separator":
                    stems = self._separate_with_audio_separator(
                        audio, sample_rate, model, report
                    )
                else:
                    stems = self._separate_with_standalone_demucs(
                        audio, sample_rate, model, device, report
                    )

                # Store in cache (evict oldest if full)
                if len(self._stem_cache) >= self._stem_cache_max:
                    oldest_key = next(iter(self._stem_cache))
                    del self._stem_cache[oldest_key]
                    logger.info(f"Evicted stem cache entry {oldest_key[:8]}…")
                self._stem_cache[cache_key] = {"stems": stems, "sample_rate": sample_rate}
                logger.info(f"Cached stems for key {cache_key[:8]}…")

            logger.info(f"Stem separation returned {len(stems)} stems: {list(stems.keys())}")
            for sname, sdata in stems.items():
                logger.info(f"  Stem '{sname}': shape={sdata.shape}, max={np.max(np.abs(sdata)):.6f}")

            # Safety fallback: if no stems extracted, use simple EQ mode
            if not stems:
                logger.warning("No stems extracted — falling back to simple EQ mode")
                report(0.4, "No stems found, applying simple EQ…")
                return apply_eq(audio, sample_rate,
                               params.get("warmth", 0.3) * enhancement_level,
                               params.get("clarity", 0.4) * enhancement_level,
                               params.get("air", 0.3) * enhancement_level,
                               params.get("dynamics", 0.3) * enhancement_level)

            report(0.4, "Enhancing stems…")

            # Enhance each stem
            enhanced_stems = {}
            vocals_level = params.get("vocals_enhance", 0.5) * enhancement_level
            drums_level = params.get("drums_enhance", 0.6) * enhancement_level
            bass_level = params.get("bass_enhance", 0.4) * enhancement_level
            other_level = params.get("other_enhance", 0.4) * enhancement_level
            clarity = params.get("clarity", 0.4) * enhancement_level
            warmth = params.get("warmth", 0.3) * enhancement_level
            air = params.get("air", 0.3) * enhancement_level
            dynamics = params.get("dynamics", 0.3) * enhancement_level

            if 'vocals' in stems:
                report(0.45, "Enhancing vocals…")
                enhanced_stems['vocals'] = _enhance_vocals(stems['vocals'], sample_rate, vocals_level, clarity, air)

            if 'drums' in stems:
                report(0.5, "Enhancing drums…")
                enhanced_stems['drums'] = _enhance_drums(stems['drums'], sample_rate, drums_level, dynamics, air)

            if 'bass' in stems:
                report(0.55, "Enhancing bass…")
                enhanced_stems['bass'] = _enhance_bass(stems['bass'], sample_rate, bass_level, warmth)

            if 'other' in stems:
                report(0.6, "Enhancing other instruments…")
                enhanced_stems['other'] = _enhance_other(stems['other'], sample_rate, other_level, clarity, warmth, air)

            report(0.65, "Remixing stems…")

            # Remix
            result = np.zeros_like(audio)
            for name, stem in enhanced_stems.items():
                if stem.shape[1] > result.shape[1]:
                    stem = stem[:, :result.shape[1]]
                elif stem.shape[1] < result.shape[1]:
                    pad_width = ((0, 0), (0, result.shape[1] - stem.shape[1]))
                    stem = np.pad(stem, pad_width, mode='constant')
                result += stem

            # Normalize
            max_val = np.max(np.abs(result))
            logger.info(f"Remix result: shape={result.shape}, max={max_val:.6f}")
            if max_val > 0.98:
                result = result * (0.98 / max_val)
            elif max_val < 0.001:
                logger.error("Remix result is near-silent! Falling back to original audio with EQ.")
                return apply_eq(audio, sample_rate,
                               params.get("warmth", 0.3) * enhancement_level,
                               params.get("clarity", 0.4) * enhancement_level,
                               params.get("air", 0.3) * enhancement_level,
                               params.get("dynamics", 0.3) * enhancement_level)

            return result

        except Exception as e:
            logger.error(f"Demucs processing failed: {e}", exc_info=True)
            report(0.1, "Demucs failed, falling back to simple mode…")
            return apply_eq(audio, sample_rate,
                           params.get("warmth", 0.3) * enhancement_level,
                           params.get("clarity", 0.4) * enhancement_level,
                           params.get("air", 0.3) * enhancement_level,
                           params.get("dynamics", 0.3) * enhancement_level)

    def _separate_with_audio_separator(self, audio: np.ndarray, sample_rate: int,
                                        separator, report) -> Dict[str, np.ndarray]:
        """Separate stems using audio_separator (writes temp files, reads back)."""
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp(prefix="ace_enhance_")
        try:
            # Write source audio to a temp WAV
            src_path = os.path.join(tmp_dir, "source.wav")
            sf.write(src_path, audio.T, sample_rate)  # [channels, samples] → [samples, channels]

            report(0.15, "Separating stems with audio_separator…")

            # IMPORTANT: audio_separator's internal handler captures output_dir
            # during load_model(), so we MUST set output_dir and re-call
            # load_model() before every separate() — same pattern as stem_service.py.
            separator.output_dir = tmp_dir
            model_name = self._demucs_model_name or "BS-Roformer-SW.ckpt"
            with _float32_default_dtype():
                separator.load_model(model_filename=model_name)
                stem_files = separator.separate(src_path)
            logger.info(f"audio_separator produced {len(stem_files)} stem files: {stem_files}")

            report(0.35, "Reading separated stems…")

            # Read stem files back as numpy arrays
            stems = {}
            stem_name_map = {
                "vocals": "vocals",
                "drums": "drums",
                "bass": "bass",
                "guitar": "other",
                "piano": "other",
                "other": "other",
            }

            for stem_path in stem_files:
                if not os.path.isabs(stem_path):
                    stem_path = os.path.join(tmp_dir, stem_path)
                if not os.path.exists(stem_path):
                    logger.warning(f"Stem file does not exist: {stem_path}")
                    continue

                basename = os.path.splitext(os.path.basename(stem_path))[0].lower()
                logger.info(f"Processing stem file: {os.path.basename(stem_path)} (basename_lower='{basename}')")

                # Try to match stem name from filename
                matched_name = None
                for key in stem_name_map:
                    if key in basename:
                        matched_name = stem_name_map[key]
                        break

                if matched_name is None:
                    logger.warning(f"Could not match stem name from filename: {basename}")
                    continue

                logger.info(f"  → Mapped to stem: '{matched_name}'")

                stem_audio, stem_sr = sf.read(stem_path, dtype='float32')
                if stem_audio.ndim == 1:
                    stem_audio = stem_audio.reshape(1, -1)
                else:
                    stem_audio = stem_audio.T  # [samples, channels] → [channels, samples]

                # Resample if stem SR differs from source (e.g. Demucs 44.1kHz vs ACE-Step 48kHz)
                if stem_sr != sample_rate:
                    logger.info(f"  Resampling stem '{matched_name}' from {stem_sr}Hz → {sample_rate}Hz")
                    if LIBROSA_AVAILABLE:
                        resampled_channels = []
                        for ch in range(stem_audio.shape[0]):
                            resampled_channels.append(
                                librosa.resample(stem_audio[ch], orig_sr=stem_sr, target_sr=sample_rate)
                            )
                        stem_audio = np.stack(resampled_channels)
                    else:
                        # Fallback: simple linear interpolation
                        ratio = sample_rate / stem_sr
                        new_len = int(stem_audio.shape[1] * ratio)
                        from scipy.interpolate import interp1d
                        x_old = np.linspace(0, 1, stem_audio.shape[1])
                        x_new = np.linspace(0, 1, new_len)
                        resampled_channels = []
                        for ch in range(stem_audio.shape[0]):
                            f = interp1d(x_old, stem_audio[ch], kind='linear')
                            resampled_channels.append(f(x_new))
                        stem_audio = np.stack(resampled_channels)

                # Accumulate if multiple files map to same stem name (e.g. guitar+piano → other)
                if matched_name in stems:
                    min_len = min(stems[matched_name].shape[1], stem_audio.shape[1])
                    stems[matched_name] = stems[matched_name][:, :min_len] + stem_audio[:, :min_len]
                else:
                    stems[matched_name] = stem_audio

            return stems

        finally:
            # Clean up temp files
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _separate_with_standalone_demucs(self, audio: np.ndarray, sample_rate: int,
                                          model, device: str, report) -> Dict[str, np.ndarray]:
        """Separate stems using standalone demucs package (tensor-based)."""
        import torch as _torch

        # Prepare tensor: [batch, channels, samples]
        audio_tensor = _torch.tensor(audio, dtype=_torch.float32).unsqueeze(0)

        # Resample if needed
        model_sr = model.samplerate
        if sample_rate != model_sr and LIBROSA_AVAILABLE:
            logger.info(f"Resampling {sample_rate}Hz → {model_sr}Hz for Demucs")
            resampled = []
            for ch in range(audio.shape[0]):
                resampled.append(librosa.resample(audio[ch], orig_sr=sample_rate, target_sr=model_sr))
            audio_tensor = _torch.tensor(np.stack(resampled), dtype=_torch.float32).unsqueeze(0)
            working_sr = model_sr
        else:
            working_sr = sample_rate

        audio_tensor = audio_tensor.to(device)

        report(0.2, "Separating stems…")

        with _torch.no_grad():
            sources = apply_model(model, audio_tensor)

        sources_np = sources.cpu().numpy()[0]
        stem_names = model.sources

        stems = {}
        for i, name in enumerate(stem_names):
            stems[name] = sources_np[i]
            # Resample back if needed
            if working_sr != sample_rate and LIBROSA_AVAILABLE:
                resampled = []
                for ch in range(stems[name].shape[0]):
                    resampled.append(librosa.resample(
                        stems[name][ch], orig_sr=working_sr, target_sr=sample_rate
                    ))
                stems[name] = np.stack(resampled)

        return stems
