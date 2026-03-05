"""HTTP route for server statistics."""

from __future__ import annotations

from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI


def register_stats_route(
    app: FastAPI,
    *,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[..., Dict[str, Any]],
    store: Any,
    queue_maxsize: int,
    initial_avg_job_seconds: float,
) -> None:
    """Register the /v1/stats endpoint."""

    @app.get("/v1/stats")
    async def get_stats(_: None = Depends(verify_api_key)):
        """Get server statistics including job store stats."""
        job_stats = store.get_stats()
        async with app.state.stats_lock:
            avg_job_seconds = getattr(app.state, "avg_job_seconds", initial_avg_job_seconds)
        return wrap_response({
            "jobs": job_stats,
            "queue_size": app.state.job_queue.qsize(),
            "queue_maxsize": queue_maxsize,
            "avg_job_seconds": avg_job_seconds,
        })
