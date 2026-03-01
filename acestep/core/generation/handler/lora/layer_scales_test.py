"""Tests for per-layer adapter scale controls."""

import unittest
from typing import Dict, Optional
from unittest.mock import patch

import torch

from acestep.core.generation.handler.lora.advanced_adapter_mixin import (
    AdvancedAdapterMixin,
    _extract_layer_index,
    _determine_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDecoder(torch.nn.Module):
    """Minimal decoder stub with 3 'layers' for testing weight merge.

    Uses an internal dict to store weights by their real dotted key names,
    so that state_dict / load_state_dict round-trips work correctly.
    """

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
    """Lightweight handler stub for testing per-layer scale APIs."""

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

    def _load_fake_adapter(self, slot: int = 0, name: str = "test_adapter"):
        """Manually inject a fake adapter for testing without real files."""
        if self._base_decoder is None:
            self._base_decoder = {
                k: v.detach().cpu().clone()
                for k, v in self.model.decoder.state_dict().items()
            }

        # Create a delta that adds 1.0 to all weights
        delta = {k: torch.ones_like(v) for k, v in self._base_decoder.items()}

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
# Tests: _extract_layer_index
# ---------------------------------------------------------------------------

class ExtractLayerIndexTests(unittest.TestCase):
    """Tests for the _extract_layer_index helper."""

    def test_standard_key(self):
        self.assertEqual(_extract_layer_index("layers.7.attn.qkv.weight"), 7)

    def test_layer_zero(self):
        self.assertEqual(_extract_layer_index("layers.0.ff.net.weight"), 0)

    def test_high_layer(self):
        self.assertEqual(_extract_layer_index("layers.23.cross_attn.k_proj.weight"), 23)

    def test_no_layers_prefix(self):
        self.assertIsNone(_extract_layer_index("scale_shift_table"))

    def test_non_numeric_after_layers(self):
        self.assertIsNone(_extract_layer_index("layers.abc.attn.weight"))

    def test_layers_at_end(self):
        self.assertIsNone(_extract_layer_index("some.layers"))

    def test_empty_string(self):
        self.assertIsNone(_extract_layer_index(""))


# ---------------------------------------------------------------------------
# Tests: set_slot_layer_scales
# ---------------------------------------------------------------------------

class SetSlotLayerScalesTests(unittest.TestCase):
    """Tests for the set_slot_layer_scales API."""

    def setUp(self):
        self.handler = _DummyHandler()
        self.handler._load_fake_adapter(slot=0)

    def test_set_layer_scales(self):
        result = self.handler.set_slot_layer_scales(0, {7: 0.5, 12: 0.0})
        self.assertIn("✅", result)
        self.assertEqual(self.handler._adapter_slots[0]["layer_scales"], {7: 0.5, 12: 0.0})

    def test_invalid_slot(self):
        result = self.handler.set_slot_layer_scales(99, {7: 0.5})
        self.assertIn("❌", result)

    def test_empty_scales_resets(self):
        self.handler.set_slot_layer_scales(0, {7: 0.5})
        result = self.handler.set_slot_layer_scales(0, {})
        self.assertIn("all=100%", result)
        self.assertEqual(self.handler._adapter_slots[0]["layer_scales"], {})

    def test_scale_clamping(self):
        self.handler.set_slot_layer_scales(0, {5: -1.0, 10: 3.0})
        scales = self.handler._adapter_slots[0]["layer_scales"]
        self.assertEqual(scales[5], 0.0)
        self.assertEqual(scales[10], 2.0)


# ---------------------------------------------------------------------------
# Tests: set_slot_layer_scale (single layer)
# ---------------------------------------------------------------------------

class SetSlotLayerScaleTests(unittest.TestCase):
    """Tests for the set_slot_layer_scale single-layer API."""

    def setUp(self):
        self.handler = _DummyHandler()
        self.handler._load_fake_adapter(slot=0)

    def test_set_single_layer(self):
        result = self.handler.set_slot_layer_scale(0, 7, 0.5)
        self.assertIn("✅", result)
        self.assertIn("layer 7", result)
        self.assertEqual(self.handler._adapter_slots[0]["layer_scales"], {7: 0.5})

    def test_setting_to_1_removes(self):
        """Setting scale to 1.0 should remove the entry (restore default)."""
        self.handler.set_slot_layer_scale(0, 7, 0.5)
        self.handler.set_slot_layer_scale(0, 7, 1.0)
        self.assertEqual(self.handler._adapter_slots[0]["layer_scales"], {})

    def test_invalid_slot(self):
        result = self.handler.set_slot_layer_scale(99, 7, 0.5)
        self.assertIn("❌", result)

    def test_multiple_layers_independently(self):
        self.handler.set_slot_layer_scale(0, 5, 0.3)
        self.handler.set_slot_layer_scale(0, 12, 0.8)
        scales = self.handler._adapter_slots[0]["layer_scales"]
        self.assertEqual(scales[5], 0.3)
        self.assertEqual(scales[12], 0.8)


# ---------------------------------------------------------------------------
# Tests: Weight merge with layer scales
# ---------------------------------------------------------------------------

class LayerScaleMergeTests(unittest.TestCase):
    """Tests that layer scales actually affect weight merging."""

    def setUp(self):
        self.handler = _DummyHandler()
        self.handler._load_fake_adapter(slot=0)

    def test_no_layer_scales_matches_original_behavior(self):
        """With empty layer_scales, merged weights should equal base + delta."""
        self.handler._apply_merged_weights_with_groups()
        sd = self.handler.model.decoder.state_dict()
        for k, base_v in self.handler._base_decoder.items():
            expected = base_v.float() + 1.0  # delta is all ones
            torch.testing.assert_close(
                sd[k].float(), expected.to(sd[k].dtype).float(),
                msg=f"Mismatch on key '{k}' with no layer scales"
            )

    def test_layer_zero_disables_layer_0(self):
        """Setting layer 0 scale to 0 should leave layer 0 keys at base."""
        self.handler.set_slot_layer_scales(0, {0: 0.0})
        sd = self.handler.model.decoder.state_dict()

        # Layer 0 keys should be unmodified (base only)
        for k, base_v in self.handler._base_decoder.items():
            idx = _extract_layer_index(k)
            if idx == 0:
                torch.testing.assert_close(
                    sd[k].float(), base_v.float(),
                    msg=f"Layer 0 key '{k}' should be at base value"
                )
            else:
                expected = base_v.float() + 1.0
                torch.testing.assert_close(
                    sd[k].float(), expected.to(sd[k].dtype).float(),
                    msg=f"Non-layer-0 key '{k}' should still have delta"
                )

    def test_partial_layer_scale(self):
        """Setting layer 1 scale to 0.5 should halve the delta for layer 1."""
        self.handler.set_slot_layer_scales(0, {1: 0.5})
        sd = self.handler.model.decoder.state_dict()

        for k, base_v in self.handler._base_decoder.items():
            idx = _extract_layer_index(k)
            if idx == 1:
                expected = base_v.float() + 0.5  # delta * 0.5
                torch.testing.assert_close(
                    sd[k].float(), expected.to(sd[k].dtype).float(),
                    msg=f"Layer 1 key '{k}' should have 50% delta"
                )

    def test_layer_scales_combined_with_group_scales(self):
        """Layer scales and group scales should multiply together."""
        # Set group self_attn to 0.5, layer 0 to 0.5 → effective = 0.25
        self.handler.set_slot_group_scales(0, self_attn_scale=0.5, cross_attn_scale=1.0, mlp_scale=1.0)
        self.handler.set_slot_layer_scales(0, {0: 0.5})
        sd = self.handler.model.decoder.state_dict()

        k = "layers.0.attn.qkv.weight"  # self_attn, layer 0
        base_v = self.handler._base_decoder[k]
        expected = base_v.float() + 0.25  # 1.0 * 0.5 (group) * 0.5 (layer) * 1.0 (delta)
        torch.testing.assert_close(
            sd[k].float(), expected.to(sd[k].dtype).float(),
            msg=f"Combined group+layer scale should be 0.25"
        )


# ---------------------------------------------------------------------------
# Tests: get_advanced_lora_status includes layer_scales
# ---------------------------------------------------------------------------

class StatusIncludesLayerScalesTests(unittest.TestCase):
    """Verify layer_scales appear in status output."""

    def test_status_includes_empty_layer_scales(self):
        handler = _DummyHandler()
        handler._load_fake_adapter(slot=0)
        status = handler.get_advanced_lora_status()
        self.assertIn("layer_scales", status["slots"][0])
        self.assertEqual(status["slots"][0]["layer_scales"], {})

    def test_status_reflects_set_layer_scales(self):
        handler = _DummyHandler()
        handler._load_fake_adapter(slot=0)
        handler.set_slot_layer_scales(0, {7: 0.0, 12: 0.5})
        status = handler.get_advanced_lora_status()
        self.assertEqual(status["slots"][0]["layer_scales"], {7: 0.0, 12: 0.5})


if __name__ == "__main__":
    unittest.main()
