"""
Compute Steering Vectors for ACE-Step

Adapted from steer-audio (https://github.com/luk-st/steer-audio)
Paper: "TADA! Tuning Audio Diffusion Models through Activation Steering"

This module computes contrastive activation addition (CAA) vectors by:
1. Running the model on positive prompts (e.g. "a pop song with piano")
2. Running the model on negative prompts (e.g. "a pop song")
3. Subtracting: steering_vector = mean(positive) - mean(negative)

The resulting .pkl files are placed in steering_vectors/ and can be loaded
via the UI's Activation Steering panel.

Usage (CLI):
    python -m acestep.compute_steering --concept piano --steps 30
    python -m acestep.compute_steering --concept custom --positive "with guitar" --negative "" --steps 30
"""

import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


# =============================================================================
# Built-in concept definitions
# =============================================================================

BASE_PROMPTS = [
    "a song", "a melody", "music", "a tune", "a track",
    "a composition", "instrumental music", "a piece of music",
    "background music", "a musical performance",
    "a pop song", "a rock song", "a jazz piece", "a classical piece",
    "electronic music", "acoustic music", "orchestral music",
    "hip hop music", "country music", "blues music",
    "folk music", "reggae music", "metal music", "punk rock",
    "dance music", "ambient music", "lofi music",
    "a ballad", "a love song", "a happy song",
    "a sad song", "energetic music", "calm music",
    "dramatic music", "cheerful music", "melancholic music",
    "aggressive music", "gentle music", "powerful music",
    "soft music", "loud music", "rhythmic music",
    "harmonious music", "simple music", "complex music",
    "minimalist music", "funky music", "groovy music",
    "cinematic music", "a soundtrack",
]


def _make_concept(positive_template: str, negative_template: str,
                  lyrics: str = "[inst]") -> Callable:
    """Create a concept factory from templates.

    Templates use {p} as the placeholder for base prompts.
    Example: positive_template="{p} with piano", negative_template="{p}"
    """
    def factory():
        pos = [positive_template.format(p=p) for p in BASE_PROMPTS]
        neg = [negative_template.format(p=p) for p in BASE_PROMPTS]
        return pos, neg, lyrics
    return factory


BUILTIN_CONCEPTS: Dict[str, Callable] = {
    "piano":         _make_concept("{p} with piano", "{p}"),
    "drums":         _make_concept("{p} with drums", "{p}"),
    "vocals":        _make_concept("{p} with vocals", "{p}", lyrics=""),
    "female_vocals": _make_concept("{p} with female vocals",
                                   "{p} with male vocals", lyrics=""),
    "mood":          _make_concept("a happy {p}", "a sad {p}"),
    "tempo":         _make_concept("fast {p}", "slow {p}"),
    "reverb":        _make_concept("{p} with heavy reverb", "{p} dry"),
    "bass":          _make_concept("{p} with heavy bass", "{p} without bass"),
    "guitar":        _make_concept("{p} with guitar", "{p}"),
    "synth":         _make_concept("{p} with synths", "{p}"),
}


def get_concept_prompts(
    concept: str,
    positive_template: Optional[str] = None,
    negative_template: Optional[str] = None,
    num_samples: int = 50,
    custom_base_prompts: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], str]:
    """Get prompt pairs for a concept.

    Args:
        concept: Built-in concept name or 'custom'
        positive_template: Custom positive template (uses {p} placeholder)
        negative_template: Custom negative template (uses {p} placeholder)
        num_samples: Max number of prompt pairs
        custom_base_prompts: Override BASE_PROMPTS with a custom list

    Returns:
        (positive_prompts, negative_prompts, lyrics)
    """
    prompts = custom_base_prompts if custom_base_prompts else BASE_PROMPTS

    if positive_template and negative_template:
        # Custom concept
        pos = [positive_template.format(p=p) for p in prompts[:num_samples]]
        neg = [negative_template.format(p=p) for p in prompts[:num_samples]]
        lyrics = "[inst]"
        return pos, neg, lyrics

    if concept not in BUILTIN_CONCEPTS:
        available = list(BUILTIN_CONCEPTS.keys())
        raise ValueError(f"Unknown concept '{concept}'. Available: {available}")

    # For built-in concepts, use custom base prompts if provided
    if custom_base_prompts:
        factory = BUILTIN_CONCEPTS[concept]
        pos, neg, lyrics = factory()
        # Rebuild using custom prompts instead of default BASE_PROMPTS
        concept_def = BUILTIN_CONCEPTS[concept]
        # Extract templates from the factory by inspecting the closure
        # Simpler: just pattern-match the first pos/neg to get templates
        if len(pos) > 0 and len(neg) > 0:
            # Reverse-engineer templates from first prompt pair
            first_base = BASE_PROMPTS[0]
            pos_tpl = pos[0].replace(first_base, '{p}') if first_base in pos[0] else '{p}'
            neg_tpl = neg[0].replace(first_base, '{p}') if first_base in neg[0] else '{p}'
            pos = [pos_tpl.format(p=p) for p in custom_base_prompts[:num_samples]]
            neg = [neg_tpl.format(p=p) for p in custom_base_prompts[:num_samples]]
        return pos[:num_samples], neg[:num_samples], lyrics

    pos, neg, lyrics = BUILTIN_CONCEPTS[concept]()
    return pos[:num_samples], neg[:num_samples], lyrics


# =============================================================================
# Core computation
# =============================================================================

def _collect_activations(
    handler,
    prompt: str,
    lyrics: str,
    num_steps: int,
    seed: int,
    device: str,
) -> Dict:
    """Run a single generation pass, collecting activations via VectorStore.

    Returns the VectorStore's vector_store dict (step -> layer -> activations).
    """
    from acestep.steering_controller import VectorStore, register_vector_control, unregister_vector_control

    # Create controller in collection mode (steer=False)
    controller = VectorStore(
        steer=False,
        device=device,
        save_only_cond=True,
        num_cfg_passes=1,  # Batched CFG does cond and uncond in 1 pass
    )

    # Get decoder (unwrap PEFT if needed)
    decoder = handler.model.decoder
    if hasattr(decoder, 'base_model') and hasattr(decoder.base_model, 'model'):
        decoder = decoder.base_model.model

    # Register hooks on all transformer blocks
    block_count = register_vector_control(decoder, controller, verbose=False)
    logger.debug(f"[steering] Registered {block_count} hooks on {type(decoder).__name__}")

    try:
        # Build generate_audio kwargs directly (bypassing _prepare_batch)
        kwargs = _build_generate_kwargs(handler, prompt, lyrics, num_steps, seed)
        handler.model.generate_audio(**kwargs)
    finally:
        # Always clean up hooks
        unregister_vector_control(decoder)

    return dict(controller.vector_store)


def _build_generate_kwargs(
    handler,
    prompt: str,
    lyrics: str,
    num_steps: int,
    seed: int,
) -> Dict[str, Any]:
    """Build the generate_audio kwargs from a simple prompt string.

    This replicates the essential preprocessing from the handler's
    service_generate method, but for a single prompt — using the correct
    handler attributes (handler.text_tokenizer, handler.infer_text_embeddings).
    """
    # Tokenize text prompt using the handler's tokenizer
    text_inputs = handler.text_tokenizer(
        prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=256,
    )
    text_token_ids = text_inputs.input_ids.to(handler.device)
    text_attention_mask = text_inputs.attention_mask.to(handler.device).bool()

    # Tokenize lyrics using the same tokenizer
    lyric_inputs = handler.text_tokenizer(
        lyrics if lyrics else "",
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=2048,
    )
    lyric_token_ids = lyric_inputs.input_ids.to(handler.device)
    lyric_attention_mask = lyric_inputs.attention_mask.to(handler.device).bool()

    # Get embeddings via the handler's encoder
    with handler._load_model_context("text_encoder"):
        text_hidden_states = handler.infer_text_embeddings(text_token_ids)
        lyric_hidden_states = handler.infer_lyric_embeddings(lyric_token_ids)

    # Build minimal dummy tensors for the generation call
    handler._ensure_silence_latent_on_device()

    # Derive latent dimensions from silence_latent
    latent_dim = handler.silence_latent.shape[-1]
    num_frames = 750  # ~30 seconds at 25Hz
    batch = 1

    src_latents = torch.zeros(batch, num_frames, latent_dim,
                              device=handler.device, dtype=handler.dtype)
    chunk_mask = torch.ones(batch, num_frames, latent_dim,
                            device=handler.device, dtype=handler.dtype)

    # Empty reference audio — use silence_latent like handler.infer_refer_latent does
    # when no reference audio is provided (see handler.py line 3670)
    refer_audio = handler.silence_latent[:, :num_frames, :].clone()
    refer_mask = torch.zeros(batch, device=handler.device, dtype=torch.long)

    return {
        "text_hidden_states": text_hidden_states,
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": refer_audio,
        "refer_audio_order_mask": refer_mask,
        "src_latents": src_latents,
        "chunk_masks": chunk_mask,
        "is_covers": torch.zeros(batch, device=handler.device, dtype=torch.bool),
        "silence_latent": handler.silence_latent,
        "seed": seed,
        "non_cover_text_hidden_states": None,
        "non_cover_text_attention_mask": None,
        "precomputed_lm_hints_25Hz": None,
        "audio_cover_strength": 1.0,
        "infer_method": "euler",
        "infer_steps": num_steps,
        "diffusion_guidance_sale": 3.0,
        "use_adg": False,
        "cfg_interval_start": 0.0,
        "cfg_interval_end": 1.0,
        "shift": 3.0,
        "use_pag": False,
        "pag_start": 0.0,
        "pag_end": 1.0,
        "pag_scale": 0.0,
    }


def _compute_steering_vectors(pos_stores, neg_stores) -> Dict:
    """Compute mean difference between positive and negative activation stores.

    Args:
        pos_stores: List of vector_store dicts from positive prompts
        neg_stores: List of vector_store dicts from negative prompts

    Returns:
        Steering vectors dict: step_key -> layer_name -> [normalized_vector]
    """
    steering_vectors = {}
    all_step_keys = sorted(pos_stores[0].keys())
    if not all_step_keys:
        raise ValueError("No activation data collected. Check that generation ran correctly.")

    layer_names = list(pos_stores[0][all_step_keys[0]].keys())
    logger.info(f"[compute_sv] Computing vectors over {len(all_step_keys)} steps, "
                f"{len(layer_names)} layers, {len(pos_stores)} samples")

    for step_key in all_step_keys:
        steering_vectors[step_key] = defaultdict(list)
        for layer_name in layer_names:
            # Average across all samples
            pos_vecs = [ps[step_key][layer_name][0] for ps in pos_stores
                        if step_key in ps and layer_name in ps[step_key]]
            neg_vecs = [ns[step_key][layer_name][0] for ns in neg_stores
                        if step_key in ns and layer_name in ns[step_key]]

            if not pos_vecs or not neg_vecs:
                continue

            pos_avg = np.mean(pos_vecs, axis=0)
            neg_avg = np.mean(neg_vecs, axis=0)
            sv = pos_avg - neg_avg

            # Normalize
            norm = np.linalg.norm(sv)
            if norm > 0:
                sv = sv / norm

            steering_vectors[step_key][layer_name].append(sv)

    return steering_vectors


def compute_concept(
    handler,
    concept: str,
    num_steps: int = 30,
    num_samples: int = 50,
    seed: int = 42,
    positive_template: Optional[str] = None,
    negative_template: Optional[str] = None,
    custom_base_prompts: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Compute steering vectors for a concept and save to disk.

    Args:
        handler: AceStepHandler instance (model must be loaded)
        concept: Concept name (built-in or 'custom')
        num_steps: Number of inference steps (must match usage)
        num_samples: Number of prompt pairs to use
        seed: Random seed for reproducibility
        positive_template: Custom positive template (optional)
        negative_template: Custom negative template (optional)
        progress_callback: fn(current, total, message) for progress updates

    Returns:
        Path to saved .pkl file
    """
    if handler.model is None:
        raise RuntimeError("Model not loaded. Call handler.load_model() first.")

    # Get prompts
    pos_prompts, neg_prompts, lyrics = get_concept_prompts(
        concept, positive_template, negative_template, num_samples,
        custom_base_prompts=custom_base_prompts,
    )
    total = len(pos_prompts) * 2  # pos + neg passes
    device = str(handler.device)

    logger.info(f"[compute_sv] Starting computation for '{concept}' "
                f"({len(pos_prompts)} pairs, {num_steps} steps, seed={seed})")

    pos_stores = []
    neg_stores = []
    t0 = time.time()

    with torch.inference_mode():
        with handler._load_model_context("model"):
            for i, (pp, np_) in enumerate(zip(pos_prompts, neg_prompts)):
                # Positive pass
                if progress_callback:
                    progress_callback(i * 2, total,
                                      f"Positive pass {i+1}/{len(pos_prompts)}: {pp[:60]}...")
                logger.info(f"[compute_sv] [{i+1}/{len(pos_prompts)}] Positive: {pp[:80]}")

                store = _collect_activations(handler, pp, lyrics, num_steps, seed, device)
                pos_stores.append(store)

                # Negative pass
                if progress_callback:
                    progress_callback(i * 2 + 1, total,
                                      f"Negative pass {i+1}/{len(neg_prompts)}: {np_[:60]}...")
                logger.info(f"[compute_sv] [{i+1}/{len(neg_prompts)}] Negative: {np_[:80]}")

                store = _collect_activations(handler, np_, lyrics, num_steps, seed, device)
                neg_stores.append(store)

    # Compute mean difference
    if progress_callback:
        progress_callback(total, total, "Computing steering vectors...")

    steering_vectors = _compute_steering_vectors(pos_stores, neg_stores)

    # Save
    sv_dir = os.path.join(handler._get_project_root(), "steering_vectors")
    os.makedirs(sv_dir, exist_ok=True)
    filepath = os.path.join(sv_dir, f"{concept}.pkl")

    with open(filepath, "wb") as f:
        pickle.dump(steering_vectors, f)

    # Save config alongside for reference
    config_path = os.path.join(sv_dir, f"{concept}_config.json")
    elapsed = time.time() - t0
    with open(config_path, "w") as f:
        json.dump({
            "concept": concept,
            "num_steps": num_steps,
            "num_samples": len(pos_prompts),
            "seed": seed,
            "elapsed_seconds": round(elapsed, 1),
            "positive_template": positive_template,
            "negative_template": negative_template,
            "num_step_keys": len(steering_vectors),
        }, f, indent=2)

    logger.info(f"[compute_sv] ✅ Saved '{concept}' vectors to {filepath} "
                f"({len(steering_vectors)} steps, {elapsed:.1f}s)")

    return filepath


# =============================================================================
# CLI entrypoint
# =============================================================================

def main():
    """CLI entrypoint for computing steering vectors."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute steering vectors for ACE-Step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Built-in concepts: {', '.join(BUILTIN_CONCEPTS.keys())}",
    )
    parser.add_argument("--concept", required=True,
                        help="Concept name (built-in or custom)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps (default: 30)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of prompt pairs (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--positive", type=str, default=None,
                        help="Custom positive template (use {p} for base prompt)")
    parser.add_argument("--negative", type=str, default=None,
                        help="Custom negative template (use {p} for base prompt)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model variant to load (default: auto-detect)")
    parser.add_argument("--all-defaults", action="store_true",
                        help="Compute all built-in concepts")

    args = parser.parse_args()

    # Import handler
    from acestep.handler import AceStepHandler

    print("🔧 Initializing handler...")
    handler = AceStepHandler()

    if args.model:
        handler.switch_model(args.model)

    print("📦 Loading model...")
    handler.load_model()

    concepts = list(BUILTIN_CONCEPTS.keys()) if args.all_defaults else [args.concept]

    for concept in concepts:
        print(f"\n🎯 Computing vectors for: {concept}")
        filepath = compute_concept(
            handler=handler,
            concept=concept,
            num_steps=args.steps,
            num_samples=args.samples,
            seed=args.seed,
            positive_template=args.positive if not args.all_defaults else None,
            negative_template=args.negative if not args.all_defaults else None,
            progress_callback=lambda cur, tot, msg: print(f"  [{cur}/{tot}] {msg}"),
        )
        print(f"  ✅ Saved to: {filepath}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
