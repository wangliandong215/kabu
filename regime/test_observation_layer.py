"""
Unit tests for regime/__init__.py::evaluate_and_log() -- the function
engine/runner.py actually calls once per run_once() pass. Covers the
"calculate -> attempt logging -> logging failure is a warning only ->
caller continues" contract, and that a RegimeProvider failure degrades to
None rather than raising.

Run:  python -m unittest regime.test_observation_layer -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import regime as regime_pkg
from regime.models import MarketContext, MarketRegime, REGIME_BULL, SOURCE_RULES
from regime.provider import RegimeProvider
import test_support


_CTX = MarketContext(weather_code=2, qqq_above_ma=True, drawdown_halt=False)


class _RaisingProvider(RegimeProvider):
    def evaluate(self, context):
        raise RuntimeError("provider exploded")


class _FixedProvider(RegimeProvider):
    def __init__(self, result):
        self._result = result

    def evaluate(self, context):
        return self._result


class TestEvaluateAndLog(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = regime_pkg.REGIME_LOG_PATH
        regime_pkg.REGIME_LOG_PATH = Path(self._tmpdir.name) / "regime_log.jsonl"

    def tearDown(self):
        regime_pkg.REGIME_LOG_PATH = self._orig_log_path
        self._tmpdir.cleanup()

    def test_returns_market_regime_and_writes_one_log_line(self):
        result = regime_pkg.evaluate_and_log(_CTX)
        self.assertIsInstance(result, MarketRegime)
        self.assertEqual(result.regime, REGIME_BULL)

        lines = regime_pkg.REGIME_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["regime"], REGIME_BULL)
        self.assertEqual(record["source"], SOURCE_RULES)
        self.assertIn("weather_code", record)
        self.assertIn("qqq_above_ma", record)
        self.assertIn("drawdown_halt", record)

    def test_injected_provider_overrides_config(self):
        fixed = MarketRegime(regime=REGIME_BULL, confidence=1.0, risk_level=1.0,
                             reason_codes=["INJECTED"], source="llm")
        result = regime_pkg.evaluate_and_log(_CTX, provider=_FixedProvider(fixed))
        self.assertIs(result, fixed)

    def test_provider_failure_returns_none_without_raising(self):
        result = regime_pkg.evaluate_and_log(_CTX, provider=_RaisingProvider())
        self.assertIsNone(result)
        # No log line written -- evaluation never produced a MarketRegime.
        self.assertFalse(regime_pkg.REGIME_LOG_PATH.exists())

    def test_logging_failure_does_not_prevent_returning_result(self):
        # Point the log path at a location that cannot be created (a file
        # standing where a directory is needed) to force the mkdir/open
        # inside _log() to fail -- evaluate_and_log() must still return the
        # computed MarketRegime, matching the "logging failure -> warning
        # only -> continue" contract from the user's spec.
        blocking_file = Path(self._tmpdir.name) / "blocked"
        blocking_file.write_text("not a directory", encoding="utf-8")
        regime_pkg.REGIME_LOG_PATH = blocking_file / "regime_log.jsonl"

        result = regime_pkg.evaluate_and_log(_CTX)
        self.assertIsInstance(result, MarketRegime)
        self.assertEqual(result.regime, REGIME_BULL)

    def test_multiple_calls_append_multiple_lines(self):
        regime_pkg.evaluate_and_log(_CTX)
        regime_pkg.evaluate_and_log(_CTX)
        lines = regime_pkg.REGIME_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
