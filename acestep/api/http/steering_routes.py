"""HTTP routes for TADA activation steering."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from acestep.handler import AceStepHandler


def register_steering_routes(
    app: FastAPI,
    *,
    verify_api_key: Callable[..., Any],
    verify_token_from_request: Callable[[dict, Optional[str]], Optional[str]],
    wrap_response: Callable[..., Dict[str, Any]],
) -> None:
    """Register activation steering endpoints."""

    def _require_handler() -> AceStepHandler:
        handler: AceStepHandler = app.state.handler
        if handler is None:
            raise HTTPException(status_code=500, detail="Handler not initialized")
        return handler

    @app.get("/v1/steering/concepts")
    async def steering_concepts(_: None = Depends(verify_api_key)):
        """List available and loaded steering concepts."""
        handler = _require_handler()
        status = handler.get_steering_status()
        # Also provide built-in concept names for the UI
        try:
            from acestep.compute_steering import BUILTIN_CONCEPTS
            status["builtin_concepts"] = list(BUILTIN_CONCEPTS.keys())
        except Exception:
            status["builtin_concepts"] = []
        return wrap_response(status)

    @app.post("/v1/steering/compute")
    async def steering_compute(request: Request, authorization: Optional[str] = Header(None)):
        """Compute steering vectors for a concept (long-running, ~15-30 min)."""
        handler = _require_handler()
        if handler.model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        body = await request.json()
        verify_token_from_request(body, authorization)

        concept = body.get("concept")
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")

        num_steps = int(body.get("num_steps", 30))
        num_samples = int(body.get("num_samples", 50))
        seed = int(body.get("seed", 42))
        positive_template = body.get("positive_template")
        negative_template = body.get("negative_template")
        custom_base_prompts = body.get("custom_base_prompts")

        result = handler.compute_steering_vectors(
            concept=concept,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            positive_template=positive_template,
            negative_template=negative_template,
            custom_base_prompts=custom_base_prompts,
        )

        status = handler.get_steering_status()
        return wrap_response({**result, **status})

    @app.post("/v1/steering/load")
    async def steering_load(request: Request, authorization: Optional[str] = Header(None)):
        """Load a computed steering vector."""
        handler = _require_handler()

        body = await request.json()
        verify_token_from_request(body, authorization)

        concept = body.get("concept")
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")

        handler.load_steering_vectors(concept)
        status = handler.get_steering_status()
        return wrap_response({"message": f"Loaded '{concept}'", **status})

    @app.post("/v1/steering/unload")
    async def steering_unload(request: Request, authorization: Optional[str] = Header(None)):
        """Unload a steering vector."""
        handler = _require_handler()

        body = await request.json()
        verify_token_from_request(body, authorization)

        concept = body.get("concept")
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")

        handler.unload_steering_vectors(concept)
        status = handler.get_steering_status()
        return wrap_response({"message": f"Unloaded '{concept}'", **status})

    @app.delete("/v1/steering/concepts/{concept}")
    async def steering_delete(concept: str, request: Request, authorization: Optional[str] = Header(None)):
        """Delete a steering vector from disk and memory."""
        handler = _require_handler()
        verify_token_from_request({}, authorization)

        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")

        msg = handler.delete_steering_vectors(concept)
        status = handler.get_steering_status()
        return wrap_response({"message": msg, **status})

    @app.post("/v1/steering/config")
    async def steering_config(request: Request, authorization: Optional[str] = Header(None)):
        """Configure steering parameters (alpha, layers, timesteps)."""
        handler = _require_handler()

        body = await request.json()
        verify_token_from_request(body, authorization)

        concept = body.get("concept")
        alpha = body.get("alpha")
        layers = body.get("layers")
        timesteps = body.get("timesteps")
        if concept and concept in getattr(handler, "steering_vectors", {}):
            cfg = handler.steering_vectors[concept].get("config", {})
            if alpha is not None:
                cfg["alpha"] = float(alpha)
            if layers is not None:
                cfg["layers"] = layers
            if timesteps is not None:
                cfg["timesteps"] = timesteps
            handler.steering_vectors[concept]["config"] = cfg

        status = handler.get_steering_status()
        return wrap_response({"message": "Config updated", **status})

    @app.post("/v1/steering/enable")
    async def steering_enable(request: Request, authorization: Optional[str] = Header(None)):
        """Enable or disable activation steering."""
        handler = _require_handler()

        body = await request.json()
        verify_token_from_request(body, authorization)

        enabled = body.get("enabled", True)
        handler.enable_steering(enabled)
        status = handler.get_steering_status()
        return wrap_response({"message": f"Steering {'enabled' if enabled else 'disabled'}", **status})
