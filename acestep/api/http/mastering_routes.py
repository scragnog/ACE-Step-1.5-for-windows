"""HTTP routes for mastering preset management and re-mastering."""

from __future__ import annotations

import json
import os
import uuid
from typing import Callable, Dict

from fastapi import FastAPI, HTTPException, Request
from loguru import logger


def register_mastering_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register mastering preset and re-master endpoints."""

    def _resolve_audio_path(audio_path: str) -> str:
        """Resolve HTTP audio URL paths to actual disk paths."""
        project_root = get_project_root()
        if audio_path.startswith("/audio/"):
            return os.path.join(project_root, "ace-step-ui", "server", "public", audio_path.lstrip("/"))
        elif audio_path.startswith("/v1/audio"):
            import urllib.parse as _urlparse
            parsed = _urlparse.urlparse(audio_path)
            qs = _urlparse.parse_qs(parsed.query)
            if "path" in qs:
                return qs["path"][0]
        elif not audio_path.startswith("http") and not os.path.isabs(audio_path):
            return os.path.join(project_root, audio_path)
        return audio_path

    @app.get("/v1/mastering/presets")
    async def list_mastering_presets():
        """List all available mastering presets."""
        from acestep.core.audio.mastering import MasteringEngine
        return {"presets": MasteringEngine.list_presets()}

    @app.post("/v1/mastering/presets")
    async def save_mastering_preset(request: Request):
        """Save a custom mastering preset."""
        body = await request.json()
        name = body.get("name", "").strip()
        params = body.get("params")

        if not name:
            raise HTTPException(400, "Preset name is required")
        if not params or not isinstance(params, dict):
            raise HTTPException(400, "Preset params dict is required")

        from acestep.core.audio.mastering import MasteringEngine
        preset_id = MasteringEngine.save_preset(name, params)
        return {"id": preset_id, "name": name}

    @app.delete("/v1/mastering/presets/{preset_id}")
    async def delete_mastering_preset(preset_id: str):
        """Delete a custom mastering preset."""
        from acestep.core.audio.mastering import MasteringEngine
        deleted = MasteringEngine.delete_preset(preset_id)
        if not deleted:
            raise HTTPException(404, f"Preset '{preset_id}' not found or protected")
        return {"deleted": preset_id}

    @app.post("/v1/mastering/apply")
    async def apply_mastering(request: Request):
        """Re-master an audio file with given parameters.

        Request body:
            audio_path: Path/URL to the original (unmastered) audio file
            mastering_params: Dict of mastering parameters to apply
        """
        body = await request.json()
        audio_path = body.get("audio_path", "")
        mastering_params = body.get("mastering_params")

        if not audio_path:
            raise HTTPException(400, "audio_path is required")

        audio_path = _resolve_audio_path(audio_path)
        if not os.path.isfile(audio_path):
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        try:
            import numpy as np
            import soundfile as sf
            from acestep.core.audio.mastering import MasteringEngine

            # Load original audio
            audio_data, sample_rate = sf.read(audio_path, dtype="float32")
            # sf.read returns [samples, channels], we need [channels, samples]
            if audio_data.ndim == 1:
                audio_data = np.stack([audio_data, audio_data])
            else:
                audio_data = audio_data.T

            # Apply mastering
            engine = MasteringEngine()
            mastered = engine.master(audio_data, sample_rate, params_override=mastering_params)

            # Save re-mastered file alongside the original, with a unique ID to prevent overwrites
            base, ext = os.path.splitext(audio_path)
            uid_suffix = uuid.uuid4().hex[:8]
            # If this is an _original file, swap it
            if base.endswith("_original"):
                output_path = base.replace("_original", f"_remastered_{uid_suffix}") + ext
            else:
                output_path = base + f"_remastered_{uid_suffix}" + ext

            # Convert back to [samples, channels]
            sf.write(output_path, mastered.T, sample_rate)

            logger.info(f"[Mastering] Re-mastered: {output_path}")
            return {"output_path": output_path, "sample_rate": sample_rate}

        except Exception as e:
            logger.error(f"[Mastering] Re-master failed: {e}", exc_info=True)
            raise HTTPException(500, f"Re-mastering failed: {e}")
