"""Tests for temporal adapter schedule and per-step re-merge."""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import torch

from acestep.core.generation.handler.lora.advanced_adapter_mixin import (
    AdvancedAdapterMixin,
    _extract_layer_index,
)
from acestep.core.generation.handler.lora.temporal_adapter_schedule import (
    AdapterSegment,
    TemporalAdapterSchedule,
)


# ---------------------------------------------------------------------------
# Reuse the FakeDecoder/DummyHandler from layer_scales_test
# ---------------------------------------------------------------------------

class _FakeDecoder(torch.nn.Module):
    _KEYS = [
        "layers.0.attn.qkv.weight",
        "layers.1.attn.qkv.weight",
        "layers.2.attn.qkv.weight",
        "layers.0.ff.net.weight",
    ]

    def __init__(self):
        super().__init__()
        self._weights: Dict[str, torch.Tensor] = {
            k: torch.zeros(4, 4) for k in self._KEYS
        }

    def state_dict(self, *args, **kwargs):
        return {k: v.clone() for k, v in self._weights.items()}

    def load_state_dict(self, state_dict, strict=True):
        for k, v in state_dict.items():
            if k in self._weights:
                self._weights[k] = v.detach().clone()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class _FakeModel:
    def __init__(self):
        self.decoder = _FakeDecoder()


class _DummyHandler(AdvancedAdapterMixin):
    def __init__(self):
        self.model = _FakeModel()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.quantization = None
        self._base_decoder = None
        self.use_lora = False
        self.lora_loaded = False
        self._adapter_slots: Dict = {}
        self._next_slot_id = 0
        self._merged_dirty = False
        self.lora_group_scales = {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0}
        self._temporal_schedule = None

    def _load_fake_adapter(self, slot: int = 0, name: str = "test", delta_val: float = 1.0):
        if self._base_decoder is None:
            self._base_decoder = {
                k: v.detach().cpu().clone()
                for k, v in self.model.decoder.state_dict().items()
            }
        delta = {k: torch.full_like(v, delta_val) for k, v in self._base_decoder.items()}
        self._adapter_slots[slot] = {
            "path": f"/fake/{name}",
            "name": name,
            "type": "peft_lora",
            "delta": delta,
            "scale": 1.0,
            "group_scales": {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0},
            "layer_scales": {},
        }
        self.use_lora = True
        self.lora_loaded = True
        self._next_slot_id = max(self._next_slot_id, slot + 1)


# ---------------------------------------------------------------------------
# Tests: TemporalAdapterSchedule interpolation
# ---------------------------------------------------------------------------

class ScheduleInterpolationTests(unittest.TestCase):
    """Tests for TemporalAdapterSchedule.get_effective_scales."""

    def test_constant_segment(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.0, end=1.0, scale=0.8)],
        })
        result = schedule.get_effective_scales(0.5)
        self.assertAlmostEqual(result[0], 0.8)

    def test_outside_all_segments(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.3, end=0.7, scale=1.0)],
        })
        self.assertAlmostEqual(schedule.get_effective_scales(0.1)[0], 0.0)
        self.assertAlmostEqual(schedule.get_effective_scales(0.9)[0], 0.0)

    def test_fade_in(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.0, end=1.0, scale=1.0, fade_in=0.2)],
        })
        # At position 0.0 → fade_in starts, scale should be 0.0
        self.assertAlmostEqual(schedule.get_effective_scales(0.0)[0], 0.0)
        # At position 0.1 → 50% through fade → scale 0.5
        self.assertAlmostEqual(schedule.get_effective_scales(0.1)[0], 0.5)
        # At position 0.2 → fade complete → scale 1.0
        self.assertAlmostEqual(schedule.get_effective_scales(0.2)[0], 1.0)
        # At position 0.5 → no fade → full scale
        self.assertAlmostEqual(schedule.get_effective_scales(0.5)[0], 1.0)

    def test_fade_out(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.0, end=1.0, scale=1.0, fade_out=0.2)],
        })
        # At position 0.5 → no fade → full scale
        self.assertAlmostEqual(schedule.get_effective_scales(0.5)[0], 1.0)
        # At position 0.9 → 50% through fade_out → scale 0.5
        self.assertAlmostEqual(schedule.get_effective_scales(0.9)[0], 0.5)
        # At position 1.0 → fade complete → scale 0.0
        self.assertAlmostEqual(schedule.get_effective_scales(1.0)[0], 0.0)

    def test_multiple_slots(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.0, end=0.5, scale=1.0)],
            1: [AdapterSegment(start=0.5, end=1.0, scale=1.0)],
        })
        scales_start = schedule.get_effective_scales(0.25)
        self.assertAlmostEqual(scales_start[0], 1.0)
        self.assertAlmostEqual(scales_start[1], 0.0)

        scales_end = schedule.get_effective_scales(0.75)
        self.assertAlmostEqual(scales_end[0], 0.0)
        self.assertAlmostEqual(scales_end[1], 1.0)

    def test_simple_switch_constructor(self):
        schedule = TemporalAdapterSchedule.simple_switch(0, 1, switch_position=0.5, crossfade=0.1)
        # At 0.25 → slot 0 full, slot 1 ramping in
        scales = schedule.get_effective_scales(0.25)
        self.assertAlmostEqual(scales[0], 1.0)
        # At 0.75 → slot 0 fading, slot 1 full
        scales = schedule.get_effective_scales(0.75)
        self.assertAlmostEqual(scales[1], 1.0)


# ---------------------------------------------------------------------------
# Tests: Schedule validation
# ---------------------------------------------------------------------------

class ScheduleValidationTests(unittest.TestCase):

    def test_valid_schedule(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.0, end=0.5, scale=1.0)],
        })
        self.assertEqual(schedule.validate(), [])

    def test_invalid_start_after_end(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.7, end=0.3, scale=1.0)],
        })
        issues = schedule.validate()
        self.assertTrue(len(issues) > 0)
        self.assertIn("start", issues[0])


# ---------------------------------------------------------------------------
# Tests: set_temporal_schedule API
# ---------------------------------------------------------------------------

class SetTemporalScheduleTests(unittest.TestCase):

    def setUp(self):
        self.handler = _DummyHandler()
        self.handler._load_fake_adapter(slot=0, name="singer_a", delta_val=1.0)
        self.handler._load_fake_adapter(slot=1, name="singer_b", delta_val=2.0)

    def test_set_valid_schedule(self):
        schedule = TemporalAdapterSchedule.simple_switch(0, 1)
        result = self.handler.set_temporal_schedule(schedule)
        self.assertIn("✅", result)
        self.assertIsNotNone(self.handler._temporal_schedule)

    def test_clear_schedule(self):
        schedule = TemporalAdapterSchedule.simple_switch(0, 1)
        self.handler.set_temporal_schedule(schedule)
        result = self.handler.set_temporal_schedule(None)
        self.assertIn("cleared", result)
        self.assertIsNone(self.handler._temporal_schedule)

    def test_invalid_schedule_rejected(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            0: [AdapterSegment(start=0.7, end=0.3, scale=1.0)],
        })
        result = self.handler.set_temporal_schedule(schedule)
        self.assertIn("❌", result)

    def test_missing_slot_rejected(self):
        schedule = TemporalAdapterSchedule(slot_segments={
            99: [AdapterSegment(start=0.0, end=1.0, scale=1.0)],
        })
        result = self.handler.set_temporal_schedule(schedule)
        self.assertIn("❌", result)
        self.assertIn("99", result)

    def test_status_reports_schedule(self):
        schedule = TemporalAdapterSchedule.simple_switch(0, 1)
        self.handler.set_temporal_schedule(schedule)
        status = self.handler.get_advanced_lora_status()
        self.assertTrue(status["temporal_schedule_active"])

    def test_status_no_schedule(self):
        status = self.handler.get_advanced_lora_status()
        self.assertFalse(status["temporal_schedule_active"])


# ---------------------------------------------------------------------------
# Tests: Temporal weight merge
# ---------------------------------------------------------------------------

class TemporalMergeTests(unittest.TestCase):

    def setUp(self):
        self.handler = _DummyHandler()
        self.handler._load_fake_adapter(slot=0, name="singer_a", delta_val=1.0)
        self.handler._load_fake_adapter(slot=1, name="singer_b", delta_val=2.0)

    def test_full_slot_0(self):
        """When slot 0 is at 1.0 and slot 1 at 0.0, should match slot 0 delta."""
        self.handler._apply_merged_weights_temporal({0: 1.0, 1: 0.0})
        sd = self.handler.model.decoder.state_dict()
        for k, base_v in self.handler._base_decoder.items():
            expected = base_v.float() + 1.0  # slot 0 delta = 1.0
            torch.testing.assert_close(sd[k].float(), expected.float())

    def test_full_slot_1(self):
        """When slot 1 is at 1.0 and slot 0 at 0.0, should match slot 1 delta."""
        self.handler._apply_merged_weights_temporal({0: 0.0, 1: 1.0})
        sd = self.handler.model.decoder.state_dict()
        for k, base_v in self.handler._base_decoder.items():
            expected = base_v.float() + 2.0  # slot 1 delta = 2.0
            torch.testing.assert_close(sd[k].float(), expected.float())

    def test_blended_50_50(self):
        """At 50/50 blend, should get half of both deltas."""
        self.handler._apply_merged_weights_temporal({0: 0.5, 1: 0.5})
        sd = self.handler.model.decoder.state_dict()
        for k, base_v in self.handler._base_decoder.items():
            expected = base_v.float() + 0.5 + 1.0  # 0.5*1.0 + 0.5*2.0
            torch.testing.assert_close(sd[k].float(), expected.float())

    def test_all_zero_restores_base(self):
        """When all scales are 0, should restore to base weights."""
        self.handler._apply_merged_weights_temporal({0: 0.0, 1: 0.0})
        sd = self.handler.model.decoder.state_dict()
        for k, base_v in self.handler._base_decoder.items():
            torch.testing.assert_close(sd[k].float(), base_v.float())


# ---------------------------------------------------------------------------
# Tests: build_temporal_step_callback
# ---------------------------------------------------------------------------

class BuildCallbackTests(unittest.TestCase):

    def test_no_schedule_returns_none(self):
        handler = _DummyHandler()
        self.assertIsNone(handler.build_temporal_step_callback())

    def test_with_schedule_returns_callable(self):
        handler = _DummyHandler()
        handler._load_fake_adapter(slot=0)
        handler._load_fake_adapter(slot=1)
        schedule = TemporalAdapterSchedule.simple_switch(0, 1)
        handler.set_temporal_schedule(schedule)
        cb = handler.build_temporal_step_callback()
        self.assertIsNotNone(cb)
        self.assertTrue(callable(cb))

    def test_callback_invokes_temporal_merge(self):
        handler = _DummyHandler()
        handler._load_fake_adapter(slot=0, name="a", delta_val=1.0)
        handler._load_fake_adapter(slot=1, name="b", delta_val=2.0)
        schedule = TemporalAdapterSchedule.simple_switch(0, 1, switch_position=0.5, crossfade=0.0)
        handler.set_temporal_schedule(schedule)
        cb = handler.build_temporal_step_callback()

        # Step 0 of 8 → position 0.0 → slot 0 should be active (scale ~1.0)
        cb(step_idx=0, t_curr=1.0, total_steps=8)
        sd_start = handler.model.decoder.state_dict()
        val_start = sd_start["layers.0.attn.qkv.weight"][0, 0].item()

        # Step 7 of 8 → position 1.0 → slot 1 should be active (scale ~1.0)
        cb(step_idx=7, t_curr=0.0, total_steps=8)
        sd_end = handler.model.decoder.state_dict()
        val_end = sd_end["layers.0.attn.qkv.weight"][0, 0].item()

        # At step 0, only slot 0 (delta=1.0) → val ≈ 1.0
        # At step 7, only slot 1 (delta=2.0) → val ≈ 2.0
        self.assertAlmostEqual(val_start, 1.0, places=3)
        self.assertAlmostEqual(val_end, 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
