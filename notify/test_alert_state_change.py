"""
Unit tests for notify/alert.py:warn_on_state_change() — push a persistent
state condition (e.g. portfolio risk tier PAUSE_NEW) only when it changes.

Run:  python -m unittest notify.test_alert_state_change -v
"""
import unittest

import notify.alert as alert
import test_support


class WarnOnStateChangeTest(unittest.TestCase):
    def setUp(self):
        self._handlers = test_support.mute_alert_file_logging()
        self._orig_push = alert._push_all
        self.pushed = []
        alert._push_all = lambda msg, prefix="": self.pushed.append(msg)
        alert._test_state.clear()

    def tearDown(self):
        alert._push_all = self._orig_push
        alert._test_state.clear()
        test_support.unmute_alert_file_logging(self._handlers)

    def test_detects_unittest_run(self):
        self.assertTrue(alert._in_test_run())

    def test_pushes_once_while_state_unchanged(self):
        for _ in range(3):
            alert.warn_on_state_change("k", "PAUSE_NEW", "paused")
        self.assertEqual(self.pushed, ["paused"])

    def test_pushes_again_on_change(self):
        alert.warn_on_state_change("k", "PAUSE_NEW", "paused")
        alert.warn_on_state_change("k", "WARNING", "warning")
        alert.warn_on_state_change("k", "PAUSE_NEW", "paused")
        self.assertEqual(self.pushed, ["paused", "warning", "paused"])

    def test_none_msg_records_without_push_and_returns_prev(self):
        self.assertIsNone(alert.warn_on_state_change("k", "PAUSE_NEW", None))
        self.assertEqual(alert.warn_on_state_change("k", "NORMAL", None), "PAUSE_NEW")
        self.assertEqual(alert.warn_on_state_change("k", "NORMAL", None), "NORMAL")
        self.assertEqual(self.pushed, [])

    def test_keys_independent(self):
        alert.warn_on_state_change("REAL", "PAUSE_NEW", "real")
        alert.warn_on_state_change("SIMULATE", "NORMAL", None)
        alert.warn_on_state_change("REAL", "PAUSE_NEW", "real")
        self.assertEqual(self.pushed, ["real"])


if __name__ == "__main__":
    unittest.main()
