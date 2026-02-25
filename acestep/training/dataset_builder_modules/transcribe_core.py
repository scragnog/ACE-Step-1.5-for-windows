"""Core transcription logic for ACE-Step Transcriber (Qwen2.5-Omni based).

Handles model loading, single-file transcription, and output parsing.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import torch
from loguru import logger


def parse_transcription(raw: str) -> Dict[str, Any]:
    """Parse transcriber output into language and lyrics.

    Expected format: ``# Languages <code> # Lyrics [Section] text ...``

    Returns:
        Dict with 'language' and 'lyrics' keys.
    """
    language = "unknown"
    lyrics = ""

    lang_match = re.search(r"#\s*Languages?\s+(\w+)", raw)
    if lang_match:
        language = lang_match.group(1).strip().lower()

    lyrics_match = re.search(r"#\s*Lyrics\s*(.*)", raw, re.DOTALL)
    if lyrics_match:
        lyrics = lyrics_match.group(1).strip()

    if not lyrics:
        lyrics = raw.strip()

    return {"language": language, "lyrics": lyrics}


def is_instrumental_output(lyrics: str) -> bool:
    """Check if transcription output indicates instrumental-only content.

    Returns True when:
    - Empty or whitespace-only output.
    - All non-whitespace text is inside ``[...]`` brackets (stage directions
      like ``[Intro]``, ``[Instrumental]``, ``[Final guitar riff]``, etc.).
    - Remaining text outside brackets is less than 5 characters.
    """
    stripped = lyrics.strip()
    if not stripped:
        return True
    cleaned = re.sub(r"\[[^\]]*\]", "", stripped)
    cleaned = re.sub(r"\s+", "", cleaned)
    return len(cleaned) < 5


def clean_repetitive_lyrics(lyrics: str) -> str:
    """Shorten lines with obvious character/word repetition in lyrics.

    Detects patterns like ``ううう...`` repeated many times or
    ``la la la la la la la ...`` that indicate model hallucination,
    and replaces them with a single instance of the repeating unit.

    Returns:
        Cleaned lyrics string with repetitive lines shortened.
    """
    cleaned_lines = []
    for line in lyrics.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if re.match(r"^\[.*\]$", stripped):
            cleaned_lines.append(line)
            continue
        text_only = re.sub(r"\[.*?\]", "", stripped).strip()
        if not text_only:
            cleaned_lines.append(line)
            continue
        pattern = _find_repeating_pattern(text_only)
        if pattern is not None:
            prefix = re.match(r"^(\[.*?\]\s*)", stripped)
            prefix_str = prefix.group(1) if prefix else ""
            replacement = prefix_str + pattern
            logger.debug("Shortened repetitive line: {} → {}", stripped[:60], replacement)
            cleaned_lines.append(replacement)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _find_repeating_pattern(text: str) -> str | None:
    """Return the repeating unit if *text* is excessively repetitive, else None.

    Handles truncated repetitions: e.g. ``ABC ABC AB`` where the last repeat
    is cut short is still detected as pattern ``ABC``.

    Word-level patterns are checked first (more meaningful for lyrics),
    then character-level patterns as fallback.
    """
    length = len(text)
    if length < 10:
        return None

    words = text.split()
    if len(words) >= 4:
        for wlen in range(1, min(6, len(words) // 2) + 1):
            wpat = words[:wlen]
            prefix_words = words[:-1]
            last_word = words[-1]
            if all(w == wpat[i % wlen] for i, w in enumerate(prefix_words)):
                expected = wpat[len(prefix_words) % wlen]
                if last_word == expected or expected.startswith(last_word):
                    full_cycles = len(words) // wlen
                    if full_cycles >= 3:
                        return " ".join(wpat)
    if len(words) >= 6:
        unique_words = set(words)
        if len(unique_words) <= 2:
            return " ".join(unique_words)

    unique_chars = set(text.replace(" ", ""))
    if len(unique_chars) <= 2 and length > 15:
        no_space = text.replace(" ", "")
        return no_space[0] if no_space else None
    for pat_len in range(1, min(20, length // 2) + 1):
        pat = text[:pat_len]
        full_repeats = length // pat_len
        remainder = length % pat_len
        matched_part = text[:pat_len * full_repeats]
        tail = text[pat_len * full_repeats:]
        if matched_part == pat * full_repeats and (remainder == 0 or pat[:remainder] == tail):
            if full_repeats >= 3:
                return pat.strip() or pat
    return None


def load_transcriber(model_path: str, device: str = "cuda"):
    """Load transcriber model and processor.

    Loads ``Qwen2_5OmniForConditionalGeneration`` then strips the audio/video
    output modules (``token2wav``, ``talker``) that are unused when
    ``return_audio=False``.  This frees ~2 GB VRAM compared to keeping them.

    Args:
        model_path: HuggingFace model ID or local path.
        device: Target device ('cuda' or 'cpu').

    Returns:
        Tuple of (model, processor).
    """
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

    logger.info("Loading transcriber model from {}", model_path)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    logger.info("Transcriber model loaded on {}", device)
    return model, processor


def transcribe_samples(
    model,
    processor,
    samples: list,
    candidate_indices: list,
    force_all: bool = False,
    return_instrumental_lyrics: bool = False,
    progress_callback=None,
    sample_callback=None,
) -> Dict[str, int]:
    """Transcribe a batch of samples, updating them in-place.

    Args:
        model: Loaded transcriber model.
        processor: Loaded transcriber processor.
        samples: Full sample list (mutable, updated in-place).
        candidate_indices: Indices into ``samples`` to transcribe.
        force_all: When True, auto-detect instrumental from output.
        return_instrumental_lyrics: When True, keep raw lyrics for instrumental
            tracks instead of replacing with ``[Instrumental]``.
        progress_callback: ``fn(idx, total, transcribed, skipped, errors)``
        sample_callback: ``fn(sample_idx, sample)`` called after each successful transcription.

    Returns:
        Dict with 'transcribed', 'instrumental', 'errors' counts.
    """
    transcribed = 0
    skipped = 0
    errors = 0
    total = len(candidate_indices)

    for idx, sample_idx in enumerate(candidate_indices):
        sample = samples[sample_idx]
        try:
            raw_output = transcribe_single(model, processor, sample.audio_path)
            parsed = parse_transcription(raw_output)
            lyrics = clean_repetitive_lyrics(parsed["lyrics"])
            lang = parsed["language"]

            detected_instrumental = is_instrumental_output(lyrics) or lang == "unknown"

            if detected_instrumental:
                sample.is_instrumental = True
                sample.language = "unknown"
                if return_instrumental_lyrics:
                    sample.lyrics = lyrics if lyrics.strip() else "[Instrumental]"
                else:
                    sample.lyrics = "[Instrumental]"
                skipped += 1
                logger.info("[Transcribe] {} → Instrumental (keep_lyrics={})", sample.filename, return_instrumental_lyrics)
            else:
                sample.lyrics = lyrics
                sample.language = lang
                sample.is_instrumental = False
                transcribed += 1
                preview = lyrics.replace("\n", " ")[:120]
                logger.info(
                    "[Transcribe] {} → lang={} | {}",
                    sample.filename, lang, preview,
                )

            if sample_callback:
                sample_callback(sample_idx, sample)

        except Exception as exc:
            logger.warning("Transcription failed for {}: {}", sample.filename, exc)
            logger.opt(exception=True).debug("Full traceback for {}", sample.filename)
            errors += 1

        if progress_callback:
            progress_callback(idx + 1, total, transcribed, skipped, errors)

    return {"transcribed": transcribed, "instrumental": skipped, "errors": errors}


def transcribe_single(model, processor, audio_path: str) -> str:
    """Transcribe a single audio file and return raw output text.

    Uses ``processor.apply_chat_template`` with ``tokenize=True`` so the
    processor handles audio loading and feature extraction internally.

    Args:
        model: Loaded transcriber model (Thinker variant).
        processor: Loaded Qwen2_5OmniProcessor.
        audio_path: Path to the audio file.

    Returns:
        Raw transcription output string.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "*Task* Transcribe this audio in detail"},
                {"type": "audio", "audio": audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    target_device = next(model.parameters()).device
    inputs = inputs.to(target_device).to(model.dtype)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs)

    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trimmed = output_ids[:, input_len:]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return result
