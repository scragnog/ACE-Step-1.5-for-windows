"""
Activation Steering Controller for ACE-Step

Ported from steer-audio (https://github.com/luk-st/steer-audio)
Paper: "TADA! Tuning Audio Diffusion Models through Activation Steering"

This module provides:
- VectorControl: Base class for activation interception
- VectorStore: Steering logic (add/remove concept vectors during diffusion)
- register_vector_control(): Hook into decoder transformer blocks
- unregister_vector_control(): Restore original forward methods
"""

import abc
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


def compute_num_cfg_passes(guidance_scale_text=0.0, guidance_scale_lyric=0.0):
    """
    Compute number of CFG passes based on guidance parameters.

    ACE-Step uses:
    - 2 passes (default): cond, uncond
    - 3 passes (double guidance): cond, cond_text_only, uncond
    """
    do_double_condition_guidance = (
        guidance_scale_text is not None
        and guidance_scale_text > 1.0
        and guidance_scale_lyric is not None
        and guidance_scale_lyric > 1.0
    )
    return 3 if do_double_condition_guidance else 2


class VectorControl(abc.ABC):
    """Base class for activation interception in transformer blocks."""

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, place_in_ace: str):
        raise NotImplementedError

    def __call__(self, vector, place_in_ace: str):
        vector = self.forward(vector, place_in_ace)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
            self.cur_step += 1
        return vector


class VectorStore(VectorControl):
    """
    Steering controller that applies precomputed concept vectors during diffusion.

    Args:
        steering_vectors: Precomputed steering vectors (dict keyed by denoising step)
        steer: Whether to apply steering (False = just collect activations)
        alpha: Forward steering strength (-100 to +100)
        beta: Backward removal strength
        steer_back: If True, remove concept instead of adding it
        device: Device for tensor operations
        save_only_cond: If True, only save conditional pass activations
        steer_mode: How to apply vectors relative to CFG passes
        num_cfg_passes: Number of CFG passes (2 or 3), None for auto-detect
    """

    VALID_MODES = [
        "cond_only",
        "uncond_only",
        "uncond_for_cond",
        "separate",
        "both_cond",
        "both_uncond",
    ]

    def __init__(
        self,
        steering_vectors=None,
        steer=True,
        alpha=10,
        beta=2,
        steer_back=False,
        device="cpu",
        save_only_cond=True,
        steer_mode="cond_only",
        num_cfg_passes=None,
    ):
        super().__init__()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.steering_vectors = steering_vectors
        self.steer = steer
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.device = device

        # CFG pass tracking
        self.num_cfg_passes = num_cfg_passes
        self.cfg_pass_count = 0
        self.save_only_cond = save_only_cond

        if steer_mode not in self.VALID_MODES:
            raise ValueError(f"steer_mode must be one of {self.VALID_MODES}, got {steer_mode}")
        self.steer_mode = steer_mode
        self.actual_denoising_step = 0

    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store()
        self.vector_store = defaultdict(dict)
        self.cfg_pass_count = 0
        self.actual_denoising_step = 0

    @staticmethod
    def get_empty_store():
        return defaultdict(list)

    def _determine_steer_key(self):
        """Determine which steering vector key to use based on mode and step."""
        if not self.steering_vectors:
            raise ValueError("Cannot steer: steering_vectors is empty")

        first_key = list(self.steering_vectors.keys())[0]

        if isinstance(first_key, tuple):
            # Keys are (denoising_step, cfg_pass)
            if len(self.steering_vectors) == 1:
                return first_key  # Turbo: single key for all steps

            if self.steer_mode in ("cond_only", "both_cond"):
                return (self.actual_denoising_step, 0)
            elif self.steer_mode in ("uncond_only", "uncond_for_cond", "both_uncond"):
                uncond_pass = (self.num_cfg_passes - 1) if self.num_cfg_passes else 1
                return (self.actual_denoising_step, uncond_pass)
            elif self.steer_mode == "separate":
                return (self.actual_denoising_step, self.cfg_pass_count)
        else:
            # Keys are denoising_step only (save_only_cond=True)
            requires_uncond = ("separate", "uncond_only", "uncond_for_cond", "both_uncond")
            if self.steer_mode in requires_uncond:
                raise ValueError(
                    f"Cannot use steer_mode='{self.steer_mode}' with steering vectors "
                    "computed with save_only_cond=True. Recompute with save_only_cond=False."
                )
            if len(self.steering_vectors) == 1:
                return first_key  # Turbo
            return self.actual_denoising_step

    def _should_steer(self):
        """Check if steering should be applied."""
        if not self.steer:
            return False
            
        # In a batched CFG implementation, there is only 1 pass per step.
        if self.num_cfg_passes == 1:
            return True

        if self.steer_mode == "cond_only":
            return self.cfg_pass_count == 0
        elif self.steer_mode == "uncond_only":
            target = (self.num_cfg_passes - 1) if self.num_cfg_passes else 1
            return self.cfg_pass_count == target
        elif self.steer_mode == "uncond_for_cond":
            return self.cfg_pass_count == 0
        elif self.steer_mode in ("separate", "both_cond", "both_uncond"):
            return True
        return False

    def forward(self, vector, place_in_ace: str):
        bz = vector.shape[0]
        if self._should_steer():
            steer_key = self._determine_steer_key()
            
            # Clamp steer_key to avoid KeyError if generation steps exceed computed steps
            if steer_key not in self.steering_vectors:
                if isinstance(steer_key, int) and self.steering_vectors:
                    steer_key = min(steer_key, max(k for k in self.steering_vectors.keys() if isinstance(k, int)))
                else:
                    return vector

            # Clamp index to available vectors (handles mismatched step lengths, e.g. cover generation)
            store = self.steering_vectors[steer_key].get(place_in_ace, [])
            if not store:
                return vector
                
            idx = min(len(self.step_store[place_in_ace]), len(store) - 1)

            if self.alpha != 0 or self.steer_back:
                # Only allocate sv_tensor when we actually need it, avoiding unnecessary bfloat16 casting
                sv = store[idx]
                sv_tensor = torch.tensor(sv, dtype=vector.dtype, device=self.device).view(1, 1, -1)
                
                # Helper to apply vector modification
                def apply_mod(v):
                    n = torch.norm(v, dim=2, keepdim=True)
                    if self.steer_back:
                        sim = torch.tensordot(v, sv_tensor, dims=([2], [2])).view(v.size()[0], v.size()[1], 1)
                        sim = torch.where(sim > 0, sim, 0)
                        v = v - (self.beta * sim) * sv_tensor.expand(v.size()[0], v.size()[1], -1)
                    else:
                        # Original steer-audio pure alpha addition + original length recovery
                        v = v + self.alpha * sv_tensor.expand(v.size()[0], v.size()[1], -1)
                        
                    # Renormalize to preserve original sequence energy (safe here as we hook attn output, not residual)
                    new_n = torch.norm(v, dim=2, keepdim=True)
                    v = v / torch.clamp(new_n, min=1e-6)
                    return v * n

                if bz > 1 and self.num_cfg_passes == 1:
                    # In batched mode (1 pass), the batch contains [cond, uncond]. 
                    # We steer cond (first half) or uncond (second half) conditionally.
                    half_bz = bz // 2
                    v_cond = vector[:half_bz]
                    v_uncond = vector[half_bz:]
                    
                    if self.steer_mode in ["cond_only", "both_cond", "separate"]:
                        v_cond = apply_mod(v_cond)
                    
                    if self.steer_mode in ["uncond_only", "uncond_for_cond", "both_uncond", "separate"]:
                        v_uncond = apply_mod(v_uncond)
                        
                    vector = torch.cat([v_cond, v_uncond], dim=0)
                else:
                    vector = apply_mod(vector)

        # Only save activations when in computation mode (steer=False)
        should_save = not self.steer
        if should_save:
            # Respect cond vs uncond saving
            if (not self.save_only_cond) or (self.cfg_pass_count == 0):
                save_vec = vector
                
                # Handle ACE-Step v1.5 batched CFG (cond and uncond in one pass)
                if bz > 1 and self.num_cfg_passes == 1:
                    half_bz = bz // 2
                    if self.save_only_cond:
                        # Cond is the first half of the batch
                        save_vec = vector[:half_bz]
                        
                vector_to_store = save_vec.data.cpu().float().numpy()
                self.step_store[place_in_ace].append(vector_to_store.mean(axis=0).mean(axis=0))

        return vector

    def between_steps(self):
        has_data = self.step_store and any(self.step_store.values())

        if self.save_only_cond:
            if has_data:
                self.vector_store[self.actual_denoising_step] = self.step_store
                
            self.actual_denoising_step += 1
            self.cfg_pass_count += 1
            if self.num_cfg_passes is not None and self.cfg_pass_count >= self.num_cfg_passes:
                self.cfg_pass_count = 0
        else:
            if has_data:
                key = (self.actual_denoising_step, self.cfg_pass_count)
                self.vector_store[key] = self.step_store
                
            self.cfg_pass_count += 1
            if self.num_cfg_passes is not None and self.cfg_pass_count >= self.num_cfg_passes:
                self.cfg_pass_count = 0
                self.actual_denoising_step += 1

        self.step_store = self.get_empty_store()


# ==================== Hook Registration ====================

# Global storage for original forward methods (keyed by id(model))
_original_forwards: Dict[int, Dict[str, Any]] = {}


def _make_hook(ctrl, place):
    """Creates a PyTorch forward hook to intercept a generic module's output."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
            
        modified = ctrl(hs, place)
        
        # If the controller didn't modify the tensor (e.g. alpha=0), 
        # return the exact original output to preserve the PyTorch graph
        if id(modified) == id(hs):
            return output
            
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


def _make_attn_hook(controller, place_in_ace):
    """Creates a PyTorch forward hook to intercept attention output."""
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            attn_out = outputs[0]
            modified = controller(attn_out, place_in_ace)
            
            # If the controller didn't modify the tensor (e.g. alpha=0), 
            # return the exact original output to preserve the PyTorch graph
            if id(modified) == id(attn_out):
                return outputs
                
            return (modified,) + outputs[1:]
        else:
            modified = controller(outputs, place_in_ace)
            
            if id(modified) == id(outputs):
                return outputs
                
            return modified
    return hook


def register_vector_control(model, controller, explicit_layers=None, verbose=False):
    """
    Register steering hooks on the decoder's transformer blocks.

    Args:
        model: The decoder model (must have .transformer_blocks, .layers, or .blocks)
        controller: VectorControl instance
        explicit_layers: Optional list of layer names to hook (e.g. ["tf6", "tf7"])
        verbose: Log registration details

    Returns:
        Total number of blocks registered
    """
    model_id = id(model)

    # Store original forwards for later restoration
    if model_id not in _original_forwards:
        _original_forwards[model_id] = {}

    # Layer class names that should be hooked
    _HOOKABLE_CLASSES = {"LinearTransformerBlock", "AceStepDiTLayer"}

    def register_recr(net_, count, place_in_ace):
        class_name = net_.__class__.__name__
        if class_name in _HOOKABLE_CLASSES:
            if explicit_layers is not None and place_in_ace not in explicit_layers:
                return count

            if class_name in ("LinearTransformerBlock", "AceStepDiTLayer"):
                # Use register_forward_hook directly on attention modules to allow hook chaining
                handles = []
                # Both generic DiT layer and legacy transformer blocks use similar structures
                has_cross = getattr(net_, 'use_cross_attention', False) or getattr(net_, 'add_cross_attention', False)
                
                if has_cross and hasattr(net_, 'cross_attn'):
                    # Intercept cross-attention output
                    h = net_.cross_attn.register_forward_hook(_make_attn_hook(controller, place_in_ace))
                    handles.append(h)
                elif hasattr(net_, 'self_attn'):
                    # Intercept DiT layer self-attention output
                    h = net_.self_attn.register_forward_hook(_make_attn_hook(controller, place_in_ace))
                    handles.append(h)
                elif hasattr(net_, 'attn'):
                    # Intercept legacy joint/self-attention output
                    h = net_.attn.register_forward_hook(_make_attn_hook(controller, place_in_ace))
                    handles.append(h)
                else:
                    # Fallback to generic hook if no specific submodules found
                    handle = net_.register_forward_hook(_make_hook(controller, place_in_ace))
                    handles.append(handle)
                
                # Store handles for removal later
                key = f"{place_in_ace}_{count}_{id(controller)}"
                _original_forwards[model_id][key] = (None, None, handles)

            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_ace)
        return count

    block_count = 0
    # Try common attribute names for the transformer layers
    for attr in ("transformer_blocks", "layers", "blocks"):
        if hasattr(model, attr):
            sub_nets = getattr(model, attr).named_children()
            break
    else:
        raise AttributeError(
            f"{type(model).__name__} has no known transformer layer attribute "
            f"(tried: transformer_blocks, layers, blocks)"
        )
    counts = {}
    for net in sub_nets:
        name = "tf" + net[0]
        count_in_block = register_recr(net[1], 0, name)
        block_count += count_in_block
        counts[name] = count_in_block

    if verbose:
        logger.info(f"[steering] Registered {block_count} blocks: {counts}")

    controller.num_att_layers = block_count
    return block_count


def unregister_vector_control(model):
    """
    Restore original forward methods, removing all steering hooks.

    Args:
        model: The decoder model that was previously hooked
    """
    model_id = id(model)
    originals = _original_forwards.pop(model_id, {})

    for key, val in originals.items():
        net_, original_forward, handles = val
        if handles is not None:
            # Remove all forward hooks
            for h in handles:
                h.remove()
        elif original_forward is not None and net_ is not None:
            # Restore original forward method (legacy pattern)
            net_.forward = original_forward

    if originals:
        logger.debug(f"[steering] Unregistered {len(originals)} hook sets")
