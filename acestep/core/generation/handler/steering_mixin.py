import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SteeringMixin:
    """Mixin for Activation Steering (TADA) support."""
    
    # Layer name mappings for UI dropdown
    _LAYER_CONFIGS = {
        "all": [f"tf{i}" for i in range(24)],
        "tf6": ["tf6"],
        "tf7": ["tf7"],
        "tf6tf7": ["tf6", "tf7"],
    }

    def get_available_steering_concepts(self) -> List[str]:
        """Scan steering_vectors/ directory for available .pkl files.

        Returns:
            List of concept names (filename without .pkl extension)
        """
        import glob
        sv_dir = os.path.join(self._get_project_root(), "steering_vectors")
        if not os.path.isdir(sv_dir):
            return []
        concepts = []
        for f in sorted(glob.glob(os.path.join(sv_dir, "*.pkl"))):
            concepts.append(os.path.splitext(os.path.basename(f))[0])
        return concepts

    def load_steering_vectors(self, concept: str) -> str:
        """Load precomputed steering vectors for a concept.

        Args:
            concept: Concept name (must match a .pkl file in steering_vectors/)

        Returns:
            Status message
        """
        import pickle

        sv_dir = os.path.join(self._get_project_root(), "steering_vectors")
        filepath = os.path.join(sv_dir, f"{concept}.pkl")

        if not os.path.exists(filepath):
            return f"❌ Steering vectors not found: {filepath}"

        try:
            with open(filepath, "rb") as f:
                vectors = pickle.load(f)
            self.steering_vectors[concept] = vectors
            # Set default config if not already configured
            if concept not in self.steering_config:
                self.steering_config[concept] = {
                    "alpha": 0.0,   # 0 = no steering by default
                    "layers": "tf7",
                    "mode": "cond_only",
                }
            logger.info(f"[steering] Loaded vectors for concept '{concept}' ({len(vectors)} steps)")
            return f"✅ Loaded steering vectors: {concept}"
        except Exception as e:
            logger.exception(f"[steering] Failed to load vectors for '{concept}'")
            return f"❌ Failed to load vectors: {e}"

    def unload_steering_vectors(self, concept: str) -> str:
        """Remove a concept's steering vectors from memory."""
        if concept in self.steering_vectors:
            del self.steering_vectors[concept]
            self.steering_config.pop(concept, None)
            logger.info(f"[steering] Unloaded vectors for '{concept}'")
            return f"✅ Unloaded: {concept}"
        return f"⚠️ Concept not loaded: {concept}"

    def delete_steering_vectors(self, concept: str) -> str:
        """Delete a concept's steering vectors from disk and memory."""
        # Unload from memory first
        self.unload_steering_vectors(concept)

        # Delete from disk
        sv_dir = os.path.join(self._get_project_root(), "steering_vectors")
        filepath = os.path.join(sv_dir, f"{concept}.pkl")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"[steering] Deleted vectors file for '{concept}'")
                return f"✅ Deleted: {concept}"
            except Exception as e:
                logger.exception(f"[steering] Failed to delete vectors file for '{concept}'")
                return f"❌ Failed to delete: {e}"
        return f"⚠️ Concept file not found: {concept}"

    def set_steering_config(self, concept: str, alpha: float = 0.0,
                            layers: str = "tf7", mode: str = "cond_only") -> str:
        """Configure steering parameters for a concept.

        Args:
            concept: Concept name (must be loaded)
            alpha: Steering strength (-100 to +100, 0 = disabled)
            layers: Target layers ("tf7", "tf6", "tf6tf7", "all")
            mode: CFG steer mode ("cond_only", "separate", "both_cond")

        Returns:
            Status message
        """
        self.steering_config[concept] = {
            "alpha": float(alpha),
            "layers": layers,
            "mode": mode,
        }
        return f"✅ {concept}: alpha={alpha}, layers={layers}, mode={mode}"

    def enable_steering(self, enabled: bool = True) -> str:
        """Master toggle for activation steering."""
        self.steering_enabled = enabled
        state = "enabled" if enabled else "disabled"
        logger.info(f"[steering] Activation steering {state}")
        return f"🎛️ Steering {state}"

    def get_steering_status(self) -> Dict[str, Any]:
        """Return current steering state for UI and debugging."""
        return {
            "enabled": getattr(self, "steering_enabled", False),
            "loaded_concepts": list(getattr(self, "steering_vectors", {}).keys()),
            "available_concepts": self.get_available_steering_concepts(),
            "config": dict(getattr(self, "steering_config", {})),
        }

    def compute_steering_vectors(
        self,
        concept: str,
        num_steps: int = 30,
        num_samples: int = 50,
        seed: int = 42,
        positive_template: str = None,
        negative_template: str = None,
        custom_base_prompts: list = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Compute steering vectors for a concept and save to disk."""
        from acestep.compute_steering import compute_concept
        import time

        if self.model is None:
            return {"status": "error", "message": "Model not loaded"}

        t0 = time.time()
        try:
            filepath = compute_concept(
                handler=self,
                concept=concept,
                num_steps=num_steps,
                num_samples=num_samples,
                seed=seed,
                positive_template=positive_template,
                negative_template=negative_template,
                custom_base_prompts=custom_base_prompts,
                progress_callback=progress_callback,
            )
            elapsed = round(time.time() - t0, 1)

            # Do NOT auto-load vectors here. Leave them on disk until the user explicitly requests them.

            return {
                "status": "ok",
                "message": f"✅ Computed '{concept}' vectors ({elapsed}s)",
                "filepath": filepath,
                "elapsed_seconds": elapsed,
                "concept": concept,
            }
        except Exception as e:
            logger.exception(f"[steering] Failed to compute vectors for '{concept}'")
            return {
                "status": "error",
                "message": f"❌ Computation failed: {e}",
            }

    def _apply_steering_hooks(self):
        """Register steering hooks on the decoder before generation."""
        from acestep.steering_controller import VectorStore, register_vector_control

        if getattr(self, "model", None) is None or self.model.decoder is None:
            logger.warning("[steering] Cannot apply hooks: model not loaded")
            return

        decoder = self.model.decoder
        # Unwrap PEFT wrapper if present
        if hasattr(decoder, 'base_model') and hasattr(decoder.base_model, 'model'):
            decoder = decoder.base_model.model

        self._steering_controllers = []

        for concept, vectors in getattr(self, "steering_vectors", {}).items():
            config = getattr(self, "steering_config", {}).get(concept, {})
            alpha = config.get("alpha", 0.0)

            # Skip concepts with zero alpha
            if alpha == 0.0:
                continue

            layers_key = config.get("layers", "tf7")
            explicit_layers = self._LAYER_CONFIGS.get(layers_key, ["tf7"])
            mode = config.get("mode", "cond_only")

            controller = VectorStore(
                steering_vectors=vectors,
                steer=True,
                alpha=alpha,
                device=self.device,
                steer_mode=mode,
                save_only_cond=True,
                num_cfg_passes=1,
            )

            block_count = register_vector_control(
                decoder, controller,
                explicit_layers=explicit_layers,
                verbose=True,
            )
            self._steering_controllers.append((concept, controller, decoder))
            logger.info(f"[steering] Applied '{concept}' (alpha={alpha}, layers={layers_key}, blocks={block_count})")

    def _remove_steering_hooks(self):
        """Remove all steering hooks from the decoder after generation."""
        from acestep.steering_controller import unregister_vector_control

        for concept, controller, decoder in getattr(self, '_steering_controllers', []):
            unregister_vector_control(decoder)
            logger.debug(f"[steering] Removed hooks for '{concept}'")

        self._steering_controllers = []
