# -*- coding: utf-8 -*-
import unittest

from position_manager import llm_interface
from position_manager.models import PositionManagerDecision
from position_manager.test_position_manager import _ctx


def _decision():
    return PositionManagerDecision(
        symbol="US.TEST", action="HOLD", current_position_pct=100.0,
        target_position_pct=100.0, delta_pct=0.0, current_qty=100,
        target_qty=100, reduction_qty=0, triggered_modules=[],
        reason="no deterioration signals triggered",
    )


class TestLLMInterface(unittest.TestCase):
    def test_off_mode_returns_none_reviewer(self):
        self.assertIsNone(llm_interface.get_llm_reviewer("OFF"))

    def test_shadow_mode_returns_placeholder_reviewer(self):
        reviewer = llm_interface.get_llm_reviewer("SHADOW")
        self.assertIsNotNone(reviewer)

    def test_active_mode_returns_placeholder_reviewer(self):
        reviewer = llm_interface.get_llm_reviewer("ACTIVE")
        self.assertIsNotNone(reviewer)

    def test_off_mode_review_position_returns_none_without_constructing_reviewer(self):
        result = llm_interface.review_position(_ctx(), _decision(), mode="OFF")
        self.assertIsNone(result)

    def test_shadow_mode_placeholder_failure_is_swallowed(self):
        # The placeholder reviewer always raises NotImplementedError —
        # review_position() must catch it and degrade to None, never
        # propagate, never block the deterministic decision.
        result = llm_interface.review_position(_ctx(), _decision(), mode="SHADOW")
        self.assertIsNone(result)

    def test_active_mode_placeholder_failure_is_swallowed(self):
        result = llm_interface.review_position(_ctx(), _decision(), mode="ACTIVE")
        self.assertIsNone(result)

    def test_default_mode_reads_config(self):
        import config
        original = config.POSITION_LLM_MODE
        try:
            config.POSITION_LLM_MODE = "OFF"
            self.assertIsNone(llm_interface.review_position(_ctx(), _decision()))
        finally:
            config.POSITION_LLM_MODE = original


if __name__ == "__main__":
    unittest.main()
