"""
Audio diff utility for layer ablation experiments.

Computes the difference between a reference audio track and an ablated
version, isolating what a specific layer change contributes.
"""
import numpy as np
from pathlib import Path
from loguru import logger


def compute_audio_diff(
    reference_path: str,
    ablated_path: str,
    output_path: str,
    amplify: float = 3.0,
) -> dict:
    """
    Compute the difference between two audio files.

    Args:
        reference_path: Path to the reference audio (full adapter)
        ablated_path: Path to the ablated audio (one layer zeroed)
        output_path: Path to save the diff audio
        amplify: Amplification factor for the diff signal (default 3x)

    Returns:
        dict with keys: output_path, rms_energy, peak, duration_match
    """
    import soundfile as sf

    # Load both files
    ref_data, ref_sr = sf.read(reference_path, dtype="float32")
    abl_data, abl_sr = sf.read(ablated_path, dtype="float32")

    if ref_sr != abl_sr:
        raise ValueError(f"Sample rate mismatch: {ref_sr} vs {abl_sr}")

    # Handle mono/stereo mismatch
    if ref_data.ndim == 1:
        ref_data = ref_data[:, np.newaxis]
    if abl_data.ndim == 1:
        abl_data = abl_data[:, np.newaxis]

    # Match lengths (truncate to shorter)
    min_len = min(len(ref_data), len(abl_data))
    duration_match = abs(len(ref_data) - len(abl_data)) <= ref_sr  # within 1 second
    ref_data = ref_data[:min_len]
    abl_data = abl_data[:min_len]

    # Compute diff
    diff = ref_data - abl_data

    # Amplify the difference to make it audible
    diff = diff * amplify

    # Clip to prevent distortion
    diff = np.clip(diff, -1.0, 1.0)

    # Compute metrics
    rms_energy = float(np.sqrt(np.mean(diff ** 2)))
    peak = float(np.max(np.abs(diff)))

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), diff, ref_sr)

    logger.info(
        f"Audio diff: RMS={rms_energy:.4f}, peak={peak:.4f}, "
        f"amplify={amplify}x, saved to {output_path}"
    )

    return {
        "output_path": str(output),
        "rms_energy": rms_energy,
        "peak": peak,
        "duration_match": duration_match,
        "sample_rate": ref_sr,
        "duration_seconds": min_len / ref_sr,
    }
