"""HTTP integration tests for dataset transcription route registration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from acestep.api import train_api_models
from acestep.api.train_api_dataset_transcribe_routes import register_training_dataset_transcribe_routes


def _wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """Return API-compatible response envelope for tests."""

    return {"data": data, "code": code, "error": error}


async def _verify_api_key(authorization: str | None = Header(None)) -> None:
    """Require fixed bearer token for test requests."""

    if authorization != "Bearer test-token":
        raise HTTPException(status_code=401, detail="Unauthorized")


class _ImmediateThread:
    """Thread double that runs the target synchronously."""

    def __init__(self, *, target: Any, daemon: bool) -> None:
        """Store thread target parameters for compatibility."""

        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        """Execute the thread target immediately."""

        self._target()


class _Sample:
    """Sample test double with attributes consumed by transcribe routes."""

    def __init__(self, *, is_instrumental: bool = False) -> None:
        """Initialize deterministic sample attributes."""

        self.filename = "sample.wav"
        self.audio_path = str(Path(tempfile.gettempdir()) / "sample.wav")
        self.duration = 10.0
        self.caption = ""
        self.genre = "electronic"
        self.prompt_override = None
        self.lyrics = ""
        self.bpm = 120
        self.keyscale = "C major"
        self.timesignature = "4/4"
        self.language = "en"
        self.is_instrumental = is_instrumental
        self.labeled = True

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary payload for checkpoint/status writes."""

        return {
            "filename": self.filename,
            "audio_path": self.audio_path,
            "duration": self.duration,
            "caption": self.caption,
            "genre": self.genre,
            "prompt_override": self.prompt_override,
            "lyrics": self.lyrics,
            "bpm": self.bpm,
            "keyscale": self.keyscale,
            "timesignature": self.timesignature,
            "language": self.language,
            "is_instrumental": self.is_instrumental,
            "labeled": self.labeled,
        }


class _Metadata:
    """Metadata test double expected by transcription routes."""

    def __init__(self, *, all_instrumental: bool = False) -> None:
        """Store the dataset-wide instrumental flag."""

        self.all_instrumental = all_instrumental

    def to_dict(self) -> dict[str, Any]:
        """Return metadata payload for incremental saves."""

        return {"all_instrumental": self.all_instrumental}


class _Builder:
    """Dataset builder test double for transcription routes."""

    def __init__(self, *, samples: list[_Sample], all_instrumental: bool = False) -> None:
        """Store deterministic sample and metadata state."""

        self.metadata = _Metadata(all_instrumental=all_instrumental)
        self.samples = samples


class _RuntimeComponentManager:
    """No-op runtime manager replacement for transcription route tests."""

    def __init__(self, handler: Any, llm: Any, app_state: Any) -> None:
        """Store runtime references for compatibility."""

        self.handler = handler
        self.llm = llm
        self.app_state = app_state

    def offload_all_to_cpu(self, include_llm: bool = False) -> bool:
        """Return the previous offload flag without touching runtime state."""

        return False

    def restore(self) -> None:
        """No-op in tests."""


class TrainApiDatasetTranscribeRoutesHttpTests(unittest.TestCase):
    """HTTP tests covering extracted transcription route behavior."""

    def setUp(self) -> None:
        """Reset global task registries before each test."""

        with train_api_models._transcribe_lock:
            train_api_models._transcribe_tasks.clear()
            train_api_models._transcribe_latest_task_id = None

    def tearDown(self) -> None:
        """Reset global task registries after each test."""

        with train_api_models._transcribe_lock:
            train_api_models._transcribe_tasks.clear()
            train_api_models._transcribe_latest_task_id = None

    def _build_client(
        self,
        *,
        samples: list[_Sample],
        all_instrumental: bool = False,
        atomic_write_json: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        append_jsonl: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> TestClient:
        """Create app/client pair with lightweight dataset + runtime state."""

        app = FastAPI()
        app.state.dataset_builder = _Builder(samples=samples, all_instrumental=all_instrumental)
        app.state.dataset_json_path = "dataset.json"
        app.state.handler = SimpleNamespace(model=object(), offload_to_cpu=False)
        app.state.llm_handler = SimpleNamespace()
        register_training_dataset_transcribe_routes(
            app=app,
            verify_api_key=_verify_api_key,
            wrap_response=_wrap_response,
            atomic_write_json=atomic_write_json or (lambda _path, _payload: None),
            append_jsonl=append_jsonl or (lambda _path, _record: None),
        )
        return TestClient(app)

    @mock.patch(
        "acestep.api.train_api_dataset_transcribe_routes.RuntimeComponentManager",
        new=_RuntimeComponentManager,
    )
    @mock.patch(
        "acestep.api.train_api_dataset_transcribe_routes.threading.Thread",
        new=_ImmediateThread,
    )
    def test_transcribe_starts_task_and_preserves_sidecar_naming(self) -> None:
        """POST /v1/dataset/transcribe should retain the local JSON sidecar naming contract."""

        append_calls: list[str] = []
        write_calls: list[str] = []
        client = self._build_client(
            samples=[_Sample(is_instrumental=False)],
            append_jsonl=lambda path, _record: append_calls.append(path),
            atomic_write_json=lambda path, _payload: write_calls.append(path),
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        )
        fake_transcribe_core = types.SimpleNamespace(
            load_transcriber=lambda _model_path, device: ("model", f"processor-{device}"),
            transcribe_samples=lambda _model, _processor, samples, candidates, **kwargs: (
                kwargs["progress_callback"](1, len(candidates), 1, 0, 0),
                kwargs["sample_callback"](candidates[0], samples[candidates[0]]),
                {"transcribed": 1, "instrumental": 0, "errors": 0},
            )[-1],
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "acestep.training.dataset_builder_modules.transcribe_core": fake_transcribe_core,
            },
        ):
            response = client.post(
                "/v1/dataset/transcribe",
                json={"model_path": "demo-transcriber"},
                headers={"Authorization": "Bearer test-token"},
            )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(200, payload["code"])
        self.assertEqual("Transcription task started", payload["data"]["message"])
        self.assertEqual(["dataset_autotranscribe.jsonl"], append_calls)
        self.assertEqual(["dataset.json"], write_calls)

        status_response = client.get(
            "/v1/dataset/transcribe_status",
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(200, status_response.status_code)
        status_payload = status_response.json()
        self.assertEqual("completed", status_payload["data"]["status"])
        self.assertEqual(1, status_payload["data"]["result"]["transcribed"])

    def test_transcribe_rejects_all_instrumental_dataset_without_force_all(self) -> None:
        """POST /v1/dataset/transcribe should reject datasets marked fully instrumental by default."""

        client = self._build_client(samples=[_Sample(is_instrumental=True)], all_instrumental=True)
        response = client.post(
            "/v1/dataset/transcribe",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(400, payload["code"])
        self.assertIn("All samples marked as instrumental", payload["error"])


if __name__ == "__main__":
    unittest.main()
