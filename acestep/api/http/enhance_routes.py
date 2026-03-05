"""HTTP routes for audio enhancement (upscaling)."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger


def register_enhance_routes(
    app: FastAPI,
    *,
    get_project_root: Callable[[], str],
) -> None:
    """Register audio enhancement endpoints."""

    _enhancer = None
    _enhance_jobs: Dict[str, Dict] = {}
    _enhance_lock = Lock()
    _enhance_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enhance")

    def _get_enhancer():
        nonlocal _enhancer
        if _enhancer is None:
            from acestep.core.audio.enhancer import AudioEnhancer
            _enhancer = AudioEnhancer()
        return _enhancer

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

    @app.get("/v1/audio/enhance/available")
    async def enhance_available():
        """Check if audio enhancement is available and which features are supported."""
        try:
            enhancer = _get_enhancer()
            return enhancer.get_available_info()
        except Exception as e:
            return {"available": False, "error": str(e)}

    @app.post("/v1/audio/enhance")
    async def enhance_audio(request: Request):
        """Start an audio enhancement job. Returns a job_id for polling progress."""
        body = await request.json()
        audio_path = body.get("audio_path", "")
        params = body.get("params", {})

        if not audio_path:
            raise HTTPException(400, "audio_path is required")

        audio_path = _resolve_audio_path(audio_path)

        if not os.path.isfile(audio_path):
            raise HTTPException(404, f"Audio file not found: {audio_path}")

        job_id = str(uuid4())
        with _enhance_lock:
            _enhance_jobs[job_id] = {
                "status": "running",
                "output_path": None,
                "error": None,
                "progress": 0.0,
                "message": "Starting…",
            }

        def _run():
            # VRAM offloading for Demucs mode
            handler = getattr(app.state, "handler", None)
            offloaded_parts = []
            use_stems = params.get("use_stem_separation", False)

            if use_stems and handler and getattr(handler, "_models_loaded", False):
                import torch as _torch
                try:
                    for attr_name in ("model", "vae", "tokenizer"):
                        mod = getattr(handler, attr_name, None)
                        if mod is not None and hasattr(mod, "to"):
                            dev = next(mod.parameters()).device if hasattr(mod, "parameters") else None
                            if dev is not None and dev.type != "cpu":
                                logger.info(f"[Enhance] Offloading {attr_name} from {dev} -> cpu")
                                mod.to("cpu")
                                offloaded_parts.append((attr_name, mod, dev))
                    _torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"[Enhance] VRAM offload failed (non-fatal): {e}")

            try:
                enhancer = _get_enhancer()
                project_root = get_project_root()
                output_dir = os.path.join(project_root, "ace-step-ui", "server", "public", "audio", "enhanced")

                def _cb(pct, msg):
                    with _enhance_lock:
                        if job_id in _enhance_jobs:
                            _enhance_jobs[job_id]["progress"] = pct
                            _enhance_jobs[job_id]["message"] = msg

                output_path = enhancer.enhance(audio_path, output_dir, params, progress_callback=_cb)

                with _enhance_lock:
                    _enhance_jobs[job_id]["status"] = "complete"
                    _enhance_jobs[job_id]["progress"] = 1.0
                    _enhance_jobs[job_id]["message"] = "Done"
                    _enhance_jobs[job_id]["output_path"] = output_path
            except Exception as e:
                logger.error(f"[Enhance] Job {job_id} failed: {e}", exc_info=True)
                with _enhance_lock:
                    _enhance_jobs[job_id]["status"] = "failed"
                    _enhance_jobs[job_id]["error"] = str(e)
                    _enhance_jobs[job_id]["message"] = f"Error: {e}"
            finally:
                if offloaded_parts:
                    import torch as _torch
                    for attr_name, mod, orig_dev in offloaded_parts:
                        try:
                            logger.info(f"[Enhance] Restoring {attr_name} -> {orig_dev}")
                            mod.to(orig_dev)
                        except Exception as e:
                            logger.warning(f"[Enhance] Failed to restore {attr_name}: {e}")
                    _torch.cuda.empty_cache()

        _enhance_executor.submit(_run)
        return {"job_id": job_id, "status": "running"}

    @app.get("/v1/audio/enhance/{job_id}/progress")
    async def enhance_progress(job_id: str):
        """SSE stream of audio enhancement progress."""

        async def _event_stream():
            while True:
                with _enhance_lock:
                    job = _enhance_jobs.get(job_id)
                if job is None:
                    yield f'data: {{"type": "error", "message": "Job not found"}}\n\n'
                    return

                status = job["status"]
                if status == "complete":
                    yield f'data: {{"type": "complete", "output_path": {json.dumps(job["output_path"])}}}\n\n'
                    return
                elif status == "failed":
                    yield f'data: {{"type": "error", "message": "{job.get("error", "Unknown error")}"}}\n\n'
                    return
                else:
                    yield f'data: {{"type": "progress", "percent": {job["progress"]:.3f}, "message": "{job.get("message", "")}"}}\n\n'

                await asyncio.sleep(0.5)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/audio/enhance/{job_id}/download")
    async def enhance_download(job_id: str):
        """Download the enhanced audio file from a completed job."""
        with _enhance_lock:
            job = _enhance_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] != "complete":
            raise HTTPException(400, f"Job not complete (status: {job['status']})")

        fp = job.get("output_path")
        if not fp or not os.path.isfile(fp):
            raise HTTPException(404, "Enhanced file missing from disk")

        return FileResponse(
            fp,
            media_type="audio/wav",
            filename=os.path.basename(fp),
        )
