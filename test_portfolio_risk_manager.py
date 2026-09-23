"""
test_portfolio_risk_manager.py -- v2.10 Portfolio Risk Manager unit tests.

Pure unit tests against risk/portfolio_risk_manager.py's evaluate()/
reconcile()/plan_rebalance() using synthetic portfolio.broker_state.BrokerState
objects -- no live OpenD connection needed (mirrors the synthetic-fixture
style of test_capacity_replacement_integration.py / test_replacement_
stabilizer_integration.py, just without needing backtest_portfolio.py's
day-by-day simulation harness since these three functions are stateless).

Run:  python -m unittest test_portfolio_risk_manager -v
"""
import unittest

import config
from portfolio.broker_state import BrokerPosition, BrokerState
from risk import portfolio_risk_manager as prm
import test_support


def _bstate(positions: dict, cash: float, total_assets: float, long_mv: float) -> BrokerState:
    return BrokerState(positions=positions, cash=cash, total_assets=total_assets, long_mv=long_mv)


def _bpos(code, qty, cost_price, market_val, current_price) -> BrokerPosition:
    return BrokerPosition(code=code, qty=qty, cost_price=cost_price,
                           market_val=market_val, current_price=current_price)


class TestEvaluate(unittest.TestCase):

    def test_normal_tier_at_or_below_95pct(self):
        bs = _bstate({}, cash=50_000, total_assets=1_000_000, long_mv=950_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_NORMAL)
        self.assertAlmostEqual(a.exposure_pct, 0.95)

    def test_pause_tier_between_95_and_100pct(self):
        bs = _bstate({}, cash=2_000, total_assets=1_000_000, long_mv=980_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_PAUSE)

    def test_warning_tier_between_100_and_105pct(self):
        bs = _bstate({}, cash=-20_000, total_assets=1_000_000, long_mv=1_020_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_WARNING)

    def test_emergency_tier_above_105pct(self):
        bs = _bstate({}, cash=-80_000, total_assets=1_000_000, long_mv=1_100_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_EMERGENCY)

    def test_negative_cash_does_not_break_the_formula(self):
        # total_assets already nets out negative cash -- exposure_pct must
        # still come out correct with no special-casing.
        bs = _bstate({}, cash=-78_695.207, total_assets=1_021_018.723, long_mv=1_099_713.93)
        a = prm.evaluate(bs)
        self.assertAlmostEqual(a.exposure_pct, 1099713.93 / 1021018.723, places=6)
        self.assertEqual(a.tier, prm.TIER_EMERGENCY)

    def test_unknown_tier_on_missing_state(self):
        a = prm.evaluate(None)
        self.assertEqual(a.tier, prm.TIER_UNKNOWN)
        self.assertIsNone(a.exposure_pct)

    def test_unknown_tier_on_nonpositive_total_assets(self):
        bs = _bstate({}, cash=0, total_assets=0, long_mv=0)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_UNKNOWN)

    def test_104pct_is_warning_not_emergency(self):
        bs = _bstate({}, cash=-40_000, total_assets=1_000_000, long_mv=1_040_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_WARNING)

    def test_105pct_boundary_is_warning_not_emergency(self):
        bs = _bstate({}, cash=-50_000, total_assets=1_000_000, long_mv=1_050_000)
        a = prm.evaluate(bs)
        self.assertEqual(a.tier, prm.TIER_WARNING,
                         "exactly 105% must stay WARNING -- EMERGENCY requires strictly above it")


class TestReconcile(unittest.TestCase):

    def test_matching_qty_no_diff(self):
        bs = _bstate({"US.QQQ": _bpos("US.QQQ", 100, 700, 72_000, 720)}, 0, 1, 1)
        diffs = prm.reconcile(bs, {"US.QQQ": {"qty": 100}})
        self.assertEqual(diffs, [])

    def test_mismatched_qty_flagged_both_directions(self):
        bs = _bstate({"US.QQQ": _bpos("US.QQQ", 473, 716, 342_000, 723)}, 0, 1, 1)
        diffs = prm.reconcile(bs, {"US.QQQ": {"qty": 349}, "US.GILD": {"qty": 130}})
        codes = {d.code: d for d in diffs}
        self.assertIn("US.QQQ", codes)
        self.assertAlmostEqual(codes["US.QQQ"].diff_qty, 473 - 349)
        self.assertIn("US.GILD", codes)   # broker doesn't have it -> broker_qty=0
        self.assertAlmostEqual(codes["US.GILD"].diff_qty, 0 - 130)

    def test_known_phantom_still_returned_by_reconcile(self):
        # US.0000 (delisted EA) is in config.RECONCILIATION_IGNORE_CODES, but
        # that list is a caller-side alert-severity choice, not a reconcile()
        # filter -- every diff must still come back for the caller to log.
        bs = _bstate({"US.0000": _bpos("US.0000", 247, 206.38, 0.0, 0.0)}, 0, 1, 1)
        diffs = prm.reconcile(bs, {})
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].code, "US.0000")
        self.assertIn("US.0000", config.RECONCILIATION_IGNORE_CODES)


class TestRankSatellitesByCurrentWeakness(unittest.TestCase):
    """v2.10.1: replaced the old entry-time-total_score ranking (which
    degenerated into pure "oldest first" once every satellite scored
    100/FULL at entry -- see the 2026-08-29 rebalance preview report) with
    one built from this pass's already-computed technical data."""

    def _pos(self, entry_time="2026-01-01T00:00:00", trail_stop=None):
        return {"strategy": "combined", "entry_price": 100.0, "qty": 10,
                "trail_stop": trail_stop, "entry_time": entry_time}

    def test_sell_signal_sorts_first_regardless_of_strength(self):
        tracker_positions = {"US.A": self._pos(), "US.B": self._pos()}
        results = {
            "US.A": {"signal": "SELL", "signal_strength": 0.9, "current_price": 100.0},
            "US.B": {"signal": "BUY", "signal_strength": 0.1, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.A", "a live SELL vote must outrank even a weak BUY")

    def test_stronger_sell_conviction_ranks_before_weaker_sell(self):
        # Regression: signal_strength is "winning_votes/total_votes" -- i.e.
        # conviction IN WHICHEVER DIRECTION WON, always positive. A 75%
        # SELL (more indicators agree) is a WEAKER holding than a 50% SELL
        # (barely a majority), so it must rank first -- not last, which a
        # naive "ascending strength for everyone" sort would do (caught via
        # a live dry-run against the real account: PAYX at 50%-strength SELL
        # was ranked ahead of AMGN at 75%-strength SELL).
        tracker_positions = {"US.STRONG_SELL": self._pos(), "US.WEAK_SELL": self._pos()}
        results = {
            "US.STRONG_SELL": {"signal": "SELL", "signal_strength": 0.75, "current_price": 100.0},
            "US.WEAK_SELL":   {"signal": "SELL", "signal_strength": 0.50, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.STRONG_SELL",
                         "the more strongly-convicted SELL should be sold first")

    def test_ascending_signal_strength_among_non_sell(self):
        tracker_positions = {"US.WEAK": self._pos(), "US.STRONG": self._pos()}
        results = {
            "US.WEAK":   {"signal": "HOLD", "signal_strength": 0.2, "current_price": 100.0},
            "US.STRONG": {"signal": "HOLD", "signal_strength": 0.9, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.WEAK")

    def test_trail_stop_distance_breaks_ties(self):
        tracker_positions = {
            "US.NEAR": self._pos(trail_stop=98.0),   # 2% above its stop
            "US.FAR":  self._pos(trail_stop=50.0),   # 50% above its stop
        }
        results = {
            "US.NEAR": {"signal": "HOLD", "signal_strength": 0.5, "current_price": 100.0},
            "US.FAR":  {"signal": "HOLD", "signal_strength": 0.5, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.NEAR", "closer to its own trailing stop should rank weaker")

    def test_entry_time_is_last_resort_tiebreak_only(self):
        tracker_positions = {
            "US.OLD": self._pos(entry_time="2026-01-01T00:00:00"),
            "US.NEW": self._pos(entry_time="2026-06-01T00:00:00"),
        }
        results = {
            "US.OLD": {"signal": "HOLD", "signal_strength": 0.5, "current_price": 100.0},
            "US.NEW": {"signal": "HOLD", "signal_strength": 0.5, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.OLD", "fully tied on everything else -> oldest first")

    def test_strong_but_old_position_no_longer_beats_weak_but_new(self):
        # The exact regression this redesign fixes: an old, currently-strong
        # position must NOT outrank (for selling) a freshly weak one just
        # because it was opened first.
        tracker_positions = {
            "US.OLD_STRONG": self._pos(entry_time="2026-01-01T00:00:00"),
            "US.NEW_WEAK":   self._pos(entry_time="2026-08-01T00:00:00"),
        }
        results = {
            "US.OLD_STRONG": {"signal": "BUY", "signal_strength": 0.9, "current_price": 100.0},
            "US.NEW_WEAK":   {"signal": "SELL", "signal_strength": 0.8, "current_price": 100.0},
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[0], "US.NEW_WEAK")

    def test_missing_scan_data_sorts_last(self):
        tracker_positions = {"US.NODATA": self._pos(), "US.WEAK": self._pos()}
        results = {
            "US.WEAK": {"signal": "SELL", "signal_strength": 0.1, "current_price": 100.0},
            # US.NODATA absent -- e.g. fetch_kline failed for it this pass
        }
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertEqual(ranked[-1], "US.NODATA", "no current data must never be guessed as weakest")

    def test_core_etf_excluded(self):
        tracker_positions = {config.QQQ_CORE_CODE: {**self._pos(), "strategy": "core_etf"},
                             "US.WEAK": self._pos()}
        results = {"US.WEAK": {"signal": "SELL", "signal_strength": 0.1, "current_price": 100.0}}
        ranked = prm._rank_satellites_by_current_weakness(tracker_positions, results)
        self.assertNotIn(config.QQQ_CORE_CODE, ranked)


class TestPlanRebalance(unittest.TestCase):

    def _base_tracker_positions(self):
        return {
            config.QQQ_CORE_CODE: {"entry_price": 716.0, "qty": 473, "strategy": "core_etf"},
            "US.WEAK":   {"entry_price": 100.0, "qty": 500, "strategy": "combined",
                          "trail_stop": None, "entry_time": "2026-01-01T00:00:00"},
            "US.STRONG": {"entry_price": 100.0, "qty": 500, "strategy": "combined",
                          "trail_stop": None, "entry_time": "2026-01-01T00:00:00"},
        }

    def _base_broker_positions(self):
        return {
            config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, 473, 716.0, 342_087.79, 723.23),
            "US.WEAK":   _bpos("US.WEAK", 500, 100.0, 60_000.0, 120.0),
            "US.STRONG": _bpos("US.STRONG", 500, 100.0, 65_000.0, 130.0),
        }

    def test_no_orders_when_already_at_or_below_target(self):
        bs = _bstate(self._base_broker_positions(), cash=100_000,
                     total_assets=1_000_000, long_mv=467_087.79)
        plan = prm.plan_rebalance(bs, self._base_tracker_positions(), target_pct=0.95, results={})
        self.assertEqual(plan, [])

    def test_qqq_excess_trimmed_first_and_never_below_floor(self):
        positions = self._base_broker_positions()
        long_mv = sum(p.market_val for p in positions.values())
        total_assets = long_mv / 1.10   # ~110% exposure -> EMERGENCY tier
        bs = _bstate(positions, cash=total_assets - long_mv,
                     total_assets=total_assets, long_mv=long_mv)
        plan = prm.plan_rebalance(bs, self._base_tracker_positions(), target_pct=0.95, results={})
        self.assertTrue(plan, "expected at least one rebalance order")
        self.assertEqual(plan[0].code, config.QQQ_CORE_CODE)
        self.assertEqual(plan[0].priority, 1)
        qqq_pos = positions[config.QQQ_CORE_CODE]
        remaining_value = (qqq_pos.qty - plan[0].sell_qty) * qqq_pos.current_price
        floor_value = config.QQQ_CORE_TARGET_PCT * total_assets
        self.assertGreaterEqual(remaining_value, floor_value - qqq_pos.current_price)

    def test_weakest_signal_satellite_sold_before_strong_one(self):
        # Push exposure high enough that QQQ trim alone can't cover it, forcing
        # priority-2 to kick in -- US.WEAK (live SELL signal) must be picked
        # over US.STRONG (live BUY signal), driven by `results`, not by
        # entry-time total_score/holding-time as before this redesign.
        positions = self._base_broker_positions()
        positions[config.QQQ_CORE_CODE] = _bpos(config.QQQ_CORE_CODE, 100, 716.0, 72_300, 723.0)
        long_mv = sum(p.market_val for p in positions.values())
        # Sized so QQQ-floor trim alone can't reach the 95% target, forcing
        # priority-2 (weakest satellite) to actually fire.
        total_assets = long_mv / 1.30   # ~130% exposure
        bs = _bstate(positions, cash=total_assets - long_mv,
                     total_assets=total_assets, long_mv=long_mv)
        results = {
            "US.WEAK":   {"signal": "SELL", "signal_strength": 0.2, "current_price": 120.0},
            "US.STRONG": {"signal": "BUY", "signal_strength": 0.9, "current_price": 130.0},
        }
        plan = prm.plan_rebalance(bs, self._base_tracker_positions(), target_pct=0.95, results=results)
        codes_sold = [o.code for o in plan]
        self.assertIn("US.WEAK", codes_sold)
        self.assertNotIn("US.STRONG", codes_sold,
                          "weakest-signal satellite should be trimmed before a strong one")

    def test_min_trade_size_skips_tiny_fragments(self):
        positions = {
            config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, 350, 716.0, 253_050, 723.0),
        }
        total_assets = 250_000  # excess is tiny -- well under PORTFOLIO_REBALANCE_MIN_TRADE_USD
        bs = _bstate(positions, cash=total_assets - 253_050,
                     total_assets=total_assets, long_mv=253_050)
        tracker_positions = {config.QQQ_CORE_CODE: {"entry_price": 716.0, "qty": 350, "strategy": "core_etf"}}
        target_pct = 253_050 / total_assets - 0.0001   # excess in dollars < min trade
        plan = prm.plan_rebalance(bs, tracker_positions, target_pct=target_pct, results={})
        for order in plan:
            self.assertGreaterEqual(order.sell_qty * order.price, config.PORTFOLIO_REBALANCE_MIN_TRADE_USD)


class TestEvaluateQQQConcentration(unittest.TestCase):
    """v2.11 QQQ CORE concentration control (Layer 1) -- see config.py
    QQQ_CORE_SOFT_LIMIT_PCT/QQQ_CORE_HARD_LIMIT_PCT docstring. Deliberately
    independent of TestEvaluate above (Layer 2, total exposure): a
    QQQ-only book makes qqq_pct == exposure_pct, which is fine for the
    tier-boundary tests here, but the two independence tests at the bottom
    pad long_mv with other holdings specifically to prove the two layers
    don't leak into each other."""

    def _qqq_state(self, qqq_pct, total_assets=1_000_000.0, price=700.0, other_value=0.0):
        qqq_value = qqq_pct * total_assets
        qty = qqq_value / price
        positions = {config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, qty, price, qqq_value, price)}
        long_mv = qqq_value + other_value
        return _bstate(positions, cash=total_assets - long_mv, total_assets=total_assets, long_mv=long_mv)

    def test_25pct_is_normal(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.25))
        self.assertEqual(a.tier, prm.QQQ_TIER_NORMAL)

    def test_28pct_is_normal(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.28))
        self.assertEqual(a.tier, prm.QQQ_TIER_NORMAL)

    def test_30pct_boundary_is_still_normal(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.30))
        self.assertEqual(a.tier, prm.QQQ_TIER_NORMAL)

    def test_32pct_is_overweight(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.32))
        self.assertEqual(a.tier, prm.QQQ_TIER_OVERWEIGHT)

    def test_35pct_boundary_is_still_overweight_not_hard_limit(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.35))
        self.assertEqual(a.tier, prm.QQQ_TIER_OVERWEIGHT,
                         "exactly at the 35% line must not yet be HARD_LIMIT -- a razor-thin "
                         "breach should not be treated the same as a real overshoot")

    def test_35_3pct_is_hard_limit(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.353))
        self.assertEqual(a.tier, prm.QQQ_TIER_HARD_LIMIT)

    def test_37pct_is_hard_limit(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.37))
        self.assertEqual(a.tier, prm.QQQ_TIER_HARD_LIMIT)

    def test_40pct_is_hard_limit(self):
        a = prm.evaluate_qqq_concentration(self._qqq_state(0.40))
        self.assertEqual(a.tier, prm.QQQ_TIER_HARD_LIMIT)

    def test_unknown_on_missing_state(self):
        a = prm.evaluate_qqq_concentration(None)
        self.assertEqual(a.tier, prm.TIER_UNKNOWN)
        self.assertIsNone(a.qqq_pct)

    def test_independent_of_total_exposure_qqq_high_exposure_normal(self):
        # QQQ alone is past its own hard limit but other holdings are small
        # enough that total portfolio exposure stays comfortably under
        # 105% -- Layer 1 and Layer 2 must classify independently.
        bs = self._qqq_state(0.40, total_assets=1_000_000.0, other_value=200_000.0)  # exposure = 60%
        self.assertEqual(prm.evaluate_qqq_concentration(bs).tier, prm.QQQ_TIER_HARD_LIMIT)
        self.assertEqual(prm.evaluate(bs).tier, prm.TIER_NORMAL)

    def test_independent_of_total_exposure_qqq_normal_exposure_emergency(self):
        # QQQ itself is well inside its own limits but other holdings push
        # total exposure past 105% -- EMERGENCY must still fire regardless
        # of QQQ's own concentration tier.
        bs = self._qqq_state(0.20, total_assets=1_000_000.0, other_value=900_000.0)  # exposure = 110%
        self.assertEqual(prm.evaluate_qqq_concentration(bs).tier, prm.QQQ_TIER_NORMAL)
        self.assertEqual(prm.evaluate(bs).tier, prm.TIER_EMERGENCY)


class TestPlanQQQCoreTrim(unittest.TestCase):

    def _qqq_state(self, qqq_pct, total_assets=1_000_000.0, price=700.0):
        qqq_value = qqq_pct * total_assets
        qty = qqq_value / price
        positions = {config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, qty, price, qqq_value, price)}
        return _bstate(positions, cash=total_assets - qqq_value, total_assets=total_assets, long_mv=qqq_value)

    def test_no_trim_at_25pct(self):
        self.assertEqual(prm.plan_qqq_core_trim(self._qqq_state(0.25)), [])

    def test_no_trim_at_30pct(self):
        self.assertEqual(prm.plan_qqq_core_trim(self._qqq_state(0.30)), [])

    def test_no_trim_at_32pct_overweight(self):
        self.assertEqual(prm.plan_qqq_core_trim(self._qqq_state(0.32)), [])

    def test_no_trim_exactly_at_35pct_boundary(self):
        self.assertEqual(prm.plan_qqq_core_trim(self._qqq_state(0.35)), [],
                         "a razor-thin 35.0% breach must not trigger an oversized trim")

    def test_no_trim_at_35_3pct_stays_pure_classification_only(self):
        # plan_qqq_core_trim() itself is a sizing function and will size a
        # real trim once past the hard limit (35.3% qualifies) -- the actual
        # "don't touch anything right after deployment" protection lives at
        # the caller in engine/runner.py via config.QQQ_CORE_TRIM_AUTO_EXECUTE
        # (see engine/test_portfolio_risk_manager_gate.py), not here. This
        # test just documents that boundary.
        plan = prm.plan_qqq_core_trim(self._qqq_state(0.353))
        self.assertTrue(plan, "35.3% is past the hard limit, so a plan IS computed here -- "
                              "execution safety is a caller-side concern")

    def test_trim_fires_above_35pct(self):
        plan = prm.plan_qqq_core_trim(self._qqq_state(0.37))
        self.assertTrue(plan, "expected a trim order past the hard limit")
        self.assertEqual(plan[0].reason, "QQQ_CORE_HARD_LIMIT_TRIM")

    def test_trim_targets_gentle_32pct_not_25pct_floor(self):
        bs = self._qqq_state(0.40)
        plan = prm.plan_qqq_core_trim(bs)
        self.assertTrue(plan)
        qqq_pos = bs.positions[config.QQQ_CORE_CODE]
        remaining_value = (qqq_pos.qty - plan[0].sell_qty) * qqq_pos.current_price
        trim_target_value = config.QQQ_CORE_TRIM_TARGET_PCT * bs.total_assets
        floor_value = config.QQQ_CORE_TARGET_PCT * bs.total_assets
        self.assertGreaterEqual(remaining_value, trim_target_value - qqq_pos.current_price,
                                "must not overshoot past the ~32% trim target")
        self.assertGreater(remaining_value, floor_value,
                           "must NOT snap all the way back down to the 25% strategic floor")

    def test_min_trade_size_skips_tiny_excess(self):
        # total_assets small enough that even a hard-limit breach's excess
        # over the 32% trim target falls below PORTFOLIO_REBALANCE_MIN_TRADE_USD.
        bs = self._qqq_state(0.351, total_assets=20_000.0)
        self.assertEqual(prm.plan_qqq_core_trim(bs), [])

    def test_no_trim_on_unknown_state(self):
        self.assertEqual(prm.plan_qqq_core_trim(None), [])

    def test_trim_size_scales_with_overshoot_and_35_01pct_is_small(self):
        # v2.11.1 audit: trim USD must grow as the overshoot past the hard
        # limit grows, and a razor-thin 35.01% breach must produce a SMALL
        # trim relative to the position -- not some fixed/runaway amount.
        pcts = [0.3501, 0.351, 0.353, 0.36, 0.37, 0.40, 0.50]
        trim_fractions = []
        for pct in pcts:
            bs = self._qqq_state(pct, total_assets=10_000_000.0)  # large enough
            # that even the smallest overshoot clears PORTFOLIO_REBALANCE_MIN_TRADE_USD
            plan = prm.plan_qqq_core_trim(bs)
            self.assertTrue(plan, f"expected a trim plan at {pct:.4%}")
            qqq_pos = bs.positions[config.QQQ_CORE_CODE]
            trim_fractions.append(plan[0].sell_qty / qqq_pos.qty)

        self.assertEqual(trim_fractions, sorted(trim_fractions),
                         "trim fraction of the QQQ position must increase monotonically "
                         "with how far past the hard limit QQQ has drifted")
        self.assertLess(trim_fractions[0], 0.10,
                        "35.01% (a razor-thin breach) must trim only a small sliver "
                        "of the position, not an abnormally large sell")
        self.assertGreater(trim_fractions[-1], trim_fractions[0] * 3,
                           "50% overweight must trim proportionally much more than 35.01%")

    def test_post_trim_weight_lands_close_to_32pct_within_share_rounding(self):
        # Allow normal integer-share/price rounding error but the estimated
        # post-trim weight must not drift meaningfully away from the
        # configured ~32% trim target across a range of overshoot sizes.
        for pct in (0.3501, 0.36, 0.37, 0.40, 0.50):
            bs = self._qqq_state(pct, total_assets=1_000_000.0, price=700.0)
            plan = prm.plan_qqq_core_trim(bs)
            self.assertTrue(plan)
            qqq_pos = bs.positions[config.QQQ_CORE_CODE]
            remaining_value = (qqq_pos.qty - plan[0].sell_qty) * qqq_pos.current_price
            post_trim_pct = remaining_value / bs.total_assets
            self.assertAlmostEqual(post_trim_pct, config.QQQ_CORE_TRIM_TARGET_PCT, delta=0.005,
                                   msg=f"post-trim weight at {pct:.4%} drifted too far from "
                                       f"the ~32% target (one share's worth of rounding is "
                                       f"expected, a whole percentage point is not)")


class TestLogQQQSnapshot(unittest.TestCase):
    """v2.11.1 audit: log_qqq_snapshot() persists the fields needed to
    answer 'would this have traded, for how much, to what estimated
    resulting weight' without needing AUTO_EXECUTE on -- see engine/
    runner.py's call site right after evaluate_qqq_concentration()."""

    def setUp(self):
        self._path = prm._QQQ_RISK_LOG_PATH
        self._had_backup = self._path.exists()
        if self._had_backup:
            self._backup = self._path.read_text(encoding="utf-8")
        self._path.unlink(missing_ok=True)

    def tearDown(self):
        self._path.unlink(missing_ok=True)
        if self._had_backup:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(self._backup, encoding="utf-8")

    def _last_row(self):
        import json
        lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1])

    def test_hard_limit_with_plan_records_trim_and_post_trim_weight(self):
        bs = _bstate({config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, 529, 700.0, 370_300.0, 700.0)},
                     cash=629_700.0, total_assets=1_000_000.0, long_mv=370_300.0)
        assessment = prm.evaluate_qqq_concentration(bs)
        plan = prm.plan_qqq_core_trim(bs)
        prm.log_qqq_snapshot(assessment, plan, executed_orders=[], skip_reason=None)

        row = self._last_row()
        self.assertEqual(row["tier"], prm.QQQ_TIER_HARD_LIMIT)
        self.assertAlmostEqual(row["qqq_weight_pct"], 0.3703, places=4)
        self.assertEqual(row["qqq_shares"], 529)
        self.assertEqual(row["hard_limit_pct"], config.QQQ_CORE_HARD_LIMIT_PCT)
        self.assertEqual(row["trim_target_pct"], config.QQQ_CORE_TRIM_TARGET_PCT)
        self.assertEqual(row["trim_shares"], plan[0].sell_qty)
        self.assertAlmostEqual(row["trim_usd"], plan[0].sell_qty * plan[0].price, places=2)
        self.assertAlmostEqual(row["estimated_post_trim_weight_pct"],
                               config.QQQ_CORE_TRIM_TARGET_PCT, delta=0.005)
        self.assertFalse(row["executed"])
        self.assertIsNone(row["skip_reason"])

    def test_hard_limit_with_empty_plan_infers_below_min_trade_size(self):
        bs = self._small_hard_limit_state()
        assessment = prm.evaluate_qqq_concentration(bs)
        plan = prm.plan_qqq_core_trim(bs)
        self.assertEqual(plan, [], "fixture must exercise the min-trade-size skip path")
        prm.log_qqq_snapshot(assessment, plan, executed_orders=[], skip_reason=None)

        row = self._last_row()
        self.assertEqual(row["skip_reason"], "BELOW_MIN_TRADE_SIZE")
        self.assertIsNone(row["trim_usd"])

    def test_emergency_active_skip_reason_is_not_overridden(self):
        bs = _bstate({config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, 473, 716.0, 342_087.79, 723.23)},
                     cash=-50_000.0, total_assets=280_000.0, long_mv=342_087.79)
        assessment = prm.evaluate_qqq_concentration(bs)
        plan = prm.plan_qqq_core_trim(bs)
        self.assertTrue(plan, "fixture must be past the hard limit for this to be meaningful")
        prm.log_qqq_snapshot(assessment, plan, executed_orders=[], skip_reason="EMERGENCY_ACTIVE")

        row = self._last_row()
        self.assertEqual(row["skip_reason"], "EMERGENCY_ACTIVE")

    def test_normal_tier_records_no_trim_fields(self):
        bs = _bstate({config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, 350, 700.0, 245_000.0, 700.0)},
                     cash=755_000.0, total_assets=1_000_000.0, long_mv=245_000.0)
        assessment = prm.evaluate_qqq_concentration(bs)
        plan = prm.plan_qqq_core_trim(bs)
        prm.log_qqq_snapshot(assessment, plan, executed_orders=[], skip_reason=None)

        row = self._last_row()
        self.assertEqual(row["tier"], prm.QQQ_TIER_NORMAL)
        self.assertIsNone(row["trim_usd"])
        self.assertIsNone(row["skip_reason"])
        self.assertFalse(row["executed"])

    def _small_hard_limit_state(self):
        total_assets = 20_000.0
        qqq_value = 0.351 * total_assets
        price = 700.0
        qty = qqq_value / price
        return _bstate({config.QQQ_CORE_CODE: _bpos(config.QQQ_CORE_CODE, qty, price, qqq_value, price)},
                        cash=total_assets - qqq_value, total_assets=total_assets, long_mv=qqq_value)


class TestReconciliationDiffGrowth(unittest.TestCase):

    def setUp(self):
        self._path = prm._LAST_DIFF_PATH
        self._had_backup = self._path.exists()
        if self._had_backup:
            self._backup = self._path.read_text(encoding="utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.unlink(missing_ok=True)

    def tearDown(self):
        self._path.unlink(missing_ok=True)
        if self._had_backup:
            self._path.write_text(self._backup, encoding="utf-8")

    def test_first_observation_is_not_flagged_as_growth(self):
        diffs = [prm.ReconciliationDiff("US.QQQ", 473, 349, 124)]
        grew = prm.check_and_update_diff_growth(diffs)
        self.assertEqual(grew.get("US.QQQ"), False)

    def test_growing_gap_is_flagged(self):
        prm.check_and_update_diff_growth([prm.ReconciliationDiff("US.QQQ", 473, 349, 124)])
        grew = prm.check_and_update_diff_growth([prm.ReconciliationDiff("US.QQQ", 500, 349, 151)])
        self.assertTrue(grew.get("US.QQQ"))

    def test_shrinking_or_stable_gap_is_not_flagged(self):
        prm.check_and_update_diff_growth([prm.ReconciliationDiff("US.QQQ", 473, 349, 124)])
        grew = prm.check_and_update_diff_growth([prm.ReconciliationDiff("US.QQQ", 460, 349, 111)])
        self.assertFalse(grew.get("US.QQQ"))


class TestQQQTrimPreviewRepeat(unittest.TestCase):
    """v2.11.2 (2026-09-15) — the [预览] push de-dup that stops sell_qty's
    own price-rounding noise (46/48/47 shares three passes in a row was the
    reported symptom) from pushing a near-identical DingTalk/Telegram
    message every ~5 minutes."""

    def setUp(self):
        self._path = prm._QQQ_TRIM_PREVIEW_PATH
        self._had_backup = self._path.exists()
        if self._had_backup:
            self._backup = self._path.read_text(encoding="utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.unlink(missing_ok=True)
        self._threshold = config.QQQ_CORE_TRIM_PREVIEW_ALERT_CHANGE_SHARES

    def tearDown(self):
        self._path.unlink(missing_ok=True)
        if self._had_backup:
            self._path.write_text(self._backup, encoding="utf-8")

    def test_first_observation_alerts(self):
        self.assertTrue(prm.check_qqq_trim_preview_repeat(46))

    def test_small_noise_within_threshold_does_not_repeat(self):
        prm.check_qqq_trim_preview_repeat(46)
        self.assertFalse(prm.check_qqq_trim_preview_repeat(46 + self._threshold - 1))

    def test_change_at_or_above_threshold_repeats(self):
        prm.check_qqq_trim_preview_repeat(46)
        self.assertTrue(prm.check_qqq_trim_preview_repeat(46 + self._threshold))

    def test_compares_against_last_pushed_value_not_last_pass(self):
        prm.check_qqq_trim_preview_repeat(46)   # pushed -> baseline=46
        # Each subsequent step is individually < threshold vs the PREVIOUS
        # pass, but a per-pass comparison would drift the baseline forward
        # and never re-alert -- this asserts the comparison stays anchored
        # to the last PUSHED value (46), not the last-seen pass.
        small_step = max(1, self._threshold - 1)
        self.assertFalse(prm.check_qqq_trim_preview_repeat(46 + small_step))
        self.assertTrue(prm.check_qqq_trim_preview_repeat(46 + self._threshold))

    def test_none_clears_baseline_for_a_fresh_re_entry(self):
        prm.check_qqq_trim_preview_repeat(46)
        prm.check_qqq_trim_preview_repeat(None)   # tier fell back below hard limit
        # A fresh CORE_HARD_LIMIT episode alerts immediately even at a value
        # close to the old baseline -- it must not compare against a stale
        # number left over from an unrelated, already-ended episode.
        self.assertTrue(prm.check_qqq_trim_preview_repeat(47))

    def test_none_with_no_prior_state_is_a_safe_no_op(self):
        self.assertFalse(prm.check_qqq_trim_preview_repeat(None))
        self.assertFalse(self._path.exists())


class TestRebalanceLock(unittest.TestCase):

    def setUp(self):
        self._path = prm._LOCK_PATH
        self._path.unlink(missing_ok=True)

    def tearDown(self):
        self._path.unlink(missing_ok=True)

    def test_acquire_then_release_allows_reacquire(self):
        self.assertEqual(prm.try_acquire_rebalance_lock(), prm.LOCK_ACQUIRED)
        self.assertEqual(prm.try_acquire_rebalance_lock(), prm.LOCK_ACTIVE,
                         "a second acquire while the first is held must be refused")
        prm.release_rebalance_lock()
        self.assertEqual(prm.try_acquire_rebalance_lock(), prm.LOCK_ACQUIRED)
        prm.release_rebalance_lock()

    def test_stale_lock_is_reported_not_cleared(self):
        import json
        import time
        record = {"pid": 999999, "started_at_epoch": time.time() - config.PORTFOLIO_REBALANCE_LOCK_TIMEOUT_SEC - 1}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(record, f)
        self.assertEqual(prm.try_acquire_rebalance_lock(), prm.LOCK_STALE)
        self.assertTrue(self._path.exists(), "a stale lock must not be auto-deleted")


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()
