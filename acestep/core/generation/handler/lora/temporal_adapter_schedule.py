"""Temporal adapter schedule for per-section adapter switching during diffusion.

Defines schedule data structures and interpolation logic that control
how adapter slot scales vary over the temporal dimension of a song.
This enables use cases like "Singer A for verses, Singer B for choruses"
with smooth crossfade transitions.

All structures are opt-in. When no temporal schedule is set, adapter
behaviour is identical to the existing static weight-space merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AdapterSegment:
    """A time-region within a song where a specific adapter scale applies.

    Positions are normalised 0.0–1.0 across the full song duration.
    ``fade_in`` and ``fade_out`` define smooth crossfade widths (also 0.0–1.0)
    at the segment start and end respectively.

    When the playhead is within [start, start + fade_in] the scale ramps
    linearly from the *previous* region's value to this segment's ``scale``.
    Similarly, [end - fade_out, end] ramps from ``scale`` toward the next.
    """

    start: float
    end: float
    scale: float = 1.0
    fade_in: float = 0.0   # normalised width of entry crossfade
    fade_out: float = 0.0  # normalised width of exit crossfade


@dataclass
class TemporalAdapterSchedule:
    """Complete temporal schedule for all adapter slots.

    ``slot_segments`` maps slot IDs to ordered lists of :class:`AdapterSegment`.
    Segments within a slot must not overlap.
    """

    slot_segments: Dict[int, List[AdapterSegment]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Interpolation
    # ------------------------------------------------------------------ #

    def get_effective_scales(self, position: float) -> Dict[int, float]:
        """Return the interpolated scale for every slot at *position*.

        Args:
            position: Normalised song position in [0.0, 1.0].

        Returns:
            Dict mapping slot ID → effective scale at that position.
        """
        scales: Dict[int, float] = {}
        for slot_id, segments in self.slot_segments.items():
            scales[slot_id] = self._interpolate_slot(segments, position)
        return scales

    @staticmethod
    def _interpolate_slot(segments: List[AdapterSegment], pos: float) -> float:
        """Compute the scale for a single slot at *pos*.

        Rules:
        - If *pos* falls inside a segment with no fade, return segment.scale.
        - If *pos* is within a fade_in region, lerp from 0 → segment.scale.
        - If *pos* is within a fade_out region, lerp from segment.scale → 0.
        - If *pos* is outside all segments, return 0.0.
        """
        for seg in segments:
            if pos < seg.start or pos > seg.end:
                continue

            # Inside this segment → compute fade
            scale = seg.scale

            # Fade in
            if seg.fade_in > 0 and pos < seg.start + seg.fade_in:
                t = (pos - seg.start) / seg.fade_in
                scale *= t

            # Fade out
            if seg.fade_out > 0 and pos > seg.end - seg.fade_out:
                t = (seg.end - pos) / seg.fade_out
                scale *= t

            return scale

        # Outside all segments
        return 0.0

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> List[str]:
        """Return a list of warnings/errors (empty = valid)."""
        issues: List[str] = []
        for slot_id, segments in self.slot_segments.items():
            for i, seg in enumerate(segments):
                if seg.start >= seg.end:
                    issues.append(f"Slot {slot_id} segment {i}: start ({seg.start}) >= end ({seg.end})")
                if seg.start < 0.0 or seg.end > 1.0:
                    issues.append(f"Slot {slot_id} segment {i}: out of [0, 1] range")
                if seg.fade_in > (seg.end - seg.start):
                    issues.append(f"Slot {slot_id} segment {i}: fade_in exceeds segment length")
                if seg.fade_out > (seg.end - seg.start):
                    issues.append(f"Slot {slot_id} segment {i}: fade_out exceeds segment length")
                # Check overlap with next segment
                if i + 1 < len(segments):
                    nxt = segments[i + 1]
                    if seg.end > nxt.start:
                        issues.append(f"Slot {slot_id}: segments {i} and {i+1} overlap")
        return issues

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def simple_switch(
        cls,
        slot_a: int,
        slot_b: int,
        switch_position: float = 0.5,
        crossfade: float = 0.05,
    ) -> "TemporalAdapterSchedule":
        """Create a simple A→B switch schedule.

        Args:
            slot_a: Slot ID for the first section.
            slot_b: Slot ID for the second section.
            switch_position: Normalised position where the switch occurs.
            crossfade: Width of the crossfade region (normalised).

        Returns:
            A schedule that plays slot_a up to switch_position, then
            crossfades to slot_b.
        """
        half_fade = crossfade / 2
        return cls(slot_segments={
            slot_a: [
                AdapterSegment(
                    start=0.0,
                    end=switch_position + half_fade,
                    scale=1.0,
                    fade_in=0.0,
                    fade_out=crossfade,
                ),
            ],
            slot_b: [
                AdapterSegment(
                    start=switch_position - half_fade,
                    end=1.0,
                    scale=1.0,
                    fade_in=crossfade,
                    fade_out=0.0,
                ),
            ],
        })

    @classmethod
    def verse_chorus(
        cls,
        slot_verse: int,
        slot_chorus: int,
        chorus_start: float,
        chorus_end: float,
        crossfade: float = 0.05,
    ) -> "TemporalAdapterSchedule":
        """Create a verse→chorus→verse schedule.

        Args:
            slot_verse: Slot ID for verse sections.
            slot_chorus: Slot ID for chorus sections.
            chorus_start: Normalised start of the chorus.
            chorus_end: Normalised end of the chorus.
            crossfade: Width of crossfade at each transition.
        """
        half_fade = crossfade / 2
        return cls(slot_segments={
            slot_verse: [
                AdapterSegment(
                    start=0.0,
                    end=chorus_start + half_fade,
                    scale=1.0,
                    fade_in=0.0,
                    fade_out=crossfade,
                ),
                AdapterSegment(
                    start=chorus_end - half_fade,
                    end=1.0,
                    scale=1.0,
                    fade_in=crossfade,
                    fade_out=0.0,
                ),
            ],
            slot_chorus: [
                AdapterSegment(
                    start=chorus_start - half_fade,
                    end=chorus_end + half_fade,
                    scale=1.0,
                    fade_in=crossfade,
                    fade_out=crossfade,
                ),
            ],
        })
