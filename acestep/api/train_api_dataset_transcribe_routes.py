"""Lyrics transcription route registration for training dataset APIs."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from acestep.api import train_api_models
from acestep.api.train_api_dataset_models import TranscribeRequest, _derive_jsonl_path
from acestep.api.train_api_runtime import RuntimeComponentManager
from acestep.handler import AceStepHandler


def register_training_dataset_transcribe_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    atomic_write_json: Callable[[str, Dict[str, Any]], None],
    append_jsonl: Callable[[str, Dict[str, Any]], None],
) -> None:
    """Register async transcription routes used by training workflows."""

    @app.post("/v1/dataset/transcribe")
    async def transcribe_lyrics(request: TranscribeRequest, _: None = Depends(verify_api_key)):
        """Start async lyrics transcription for eligible dataset samples."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        if not builder.samples:
            raise HTTPException(status_code=400, detail="Dataset has no samples")
        if not request.force_all and builder.metadata.all_instrumental:
            return wrap_response(
                None,
                code=400,
                error="All samples marked as instrumental. Uncheck 'All Instrumental' or use force_all=true.",
            )

        candidates = [i for i, sample in enumerate(builder.samples) if request.force_all or not sample.is_instrumental]
        if not candidates:
            return wrap_response(
                None,
                code=400,
                error="No samples to transcribe. All samples are marked as instrumental.",
            )

        resolved_save_path = getattr(app.state, "dataset_json_path", None)
        resolved_save_path = os.path.normpath(resolved_save_path) if resolved_save_path else None
        resolved_jsonl_path = (
            _derive_jsonl_path(resolved_save_path, "_autotranscribe") if resolved_save_path else None
        )

        task_id = str(uuid4())
        with train_api_models._transcribe_lock:
            train_api_models._transcribe_tasks[task_id] = train_api_models.TranscribeTask(
                task_id=task_id,
                status="running",
                progress="Loading transcriber model...",
                current=0,
                total=len(candidates),
                created_at=time.time(),
                updated_at=time.time(),
                save_path=resolved_save_path,
            )
            train_api_models._transcribe_latest_task_id = task_id

        def run_transcription() -> None:
            import gc

            import torch

            from acestep.training.dataset_builder_modules.transcribe_core import (
                load_transcriber,
                transcribe_samples,
            )

            handler: AceStepHandler = app.state.handler
            llm = getattr(app.state, "llm_handler", None)
            mgr = RuntimeComponentManager(handler=handler, llm=llm, app_state=app.state)

            transcriber_model = None
            transcriber_processor = None
            prev_offload = mgr.offload_all_to_cpu(include_llm=True)
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                transcriber_model, transcriber_processor = load_transcriber(
                    request.model_path,
                    device=device,
                )

                def on_progress(current: int, total: int, ok: int, inst: int, err: int) -> None:
                    with train_api_models._transcribe_lock:
                        task = train_api_models._transcribe_tasks.get(task_id)
                        if task:
                            task.current = current
                            task.progress = f"Transcribing {current}/{total} (✅{ok} 🎵{inst} ❌{err})"
                            task.updated_at = time.time()

                def sample_transcribed_callback(sample_idx: int, sample: Any) -> None:
                    with train_api_models._transcribe_lock:
                        task = train_api_models._transcribe_tasks.get(task_id)
                        if task:
                            task.last_updated_index = sample_idx
                            task.last_updated_sample = sample.to_dict()
                            task.updated_at = time.time()

                    if resolved_save_path is None:
                        return
                    try:
                        if resolved_jsonl_path is not None:
                            append_jsonl(
                                resolved_jsonl_path,
                                {
                                    "ts": time.time(),
                                    "index": sample_idx,
                                    "sample": sample.to_dict(),
                                },
                            )
                        atomic_write_json(
                            resolved_save_path,
                            {
                                "metadata": builder.metadata.to_dict(),
                                "samples": [entry.to_dict() for entry in builder.samples],
                            },
                        )
                    except Exception:
                        logger.exception("Transcribe incremental save failed")

                counts = transcribe_samples(
                    transcriber_model,
                    transcriber_processor,
                    builder.samples,
                    candidates,
                    force_all=request.force_all,
                    return_instrumental_lyrics=request.return_instrumental_lyrics,
                    progress_callback=on_progress,
                    sample_callback=sample_transcribed_callback,
                )
                status_msg = (
                    f"✅ Transcription complete: {counts['transcribed']} transcribed, "
                    f"{counts['instrumental']} instrumental, {counts['errors']} errors"
                )
                with train_api_models._transcribe_lock:
                    task = train_api_models._transcribe_tasks.get(task_id)
                    if task:
                        task.status = "completed"
                        task.progress = status_msg
                        task.current = task.total
                        task.updated_at = time.time()
                        task.result = {
                            "message": status_msg,
                            "transcribed": counts["transcribed"],
                            "instrumental": counts["instrumental"],
                            "errors": counts["errors"],
                        }
            except Exception as exc:
                logger.exception("Transcription task failed")
                with train_api_models._transcribe_lock:
                    task = train_api_models._transcribe_tasks.get(task_id)
                    if task:
                        task.status = "failed"
                        task.error = str(exc)
                        task.progress = f"Failed: {exc}"
                        task.updated_at = time.time()
            finally:
                transcriber_model = None
                transcriber_processor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                handler.offload_to_cpu = prev_offload
                mgr.restore()

        threading.Thread(target=run_transcription, daemon=True).start()

        return wrap_response(
            {
                "task_id": task_id,
                "message": "Transcription task started",
                "total": len(candidates),
            }
        )

    @app.get("/v1/dataset/transcribe_status")
    async def get_transcribe_status_latest(_: None = Depends(verify_api_key)):
        """Get latest transcription task status."""

        with train_api_models._transcribe_lock:
            latest_id = train_api_models._transcribe_latest_task_id
            if latest_id is None:
                return wrap_response({"task_id": None, "status": "idle", "progress": "", "current": 0, "total": 0})

            task = train_api_models._transcribe_tasks.get(latest_id)
            if task is None:
                return wrap_response({"task_id": latest_id, "status": "idle", "progress": "", "current": 0, "total": 0})

            data: Dict[str, Any] = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "current": task.current,
                "total": task.total,
                "save_path": task.save_path,
                "last_updated_index": task.last_updated_index,
                "last_updated_sample": task.last_updated_sample,
            }
            if task.status == "completed" and task.result:
                data["result"] = task.result
            elif task.status == "failed" and task.error:
                data["error"] = task.error
            return wrap_response(data)
