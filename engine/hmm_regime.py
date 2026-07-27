# -*- coding: utf-8 -*-
"""
engine/hmm_regime.py — HMM-based Market Regime Detection (V2.7 Stage 3, Step 1).

This is the stage-2 `MLRegimeClassifier` that engine/market_regime.py's
docstring has anticipated since V2.5 ("a stage-2 unsupervised model (GMM/HMM)
can be added later as another implementation without changing any caller").
It implements the same `RegimeClassifier` ABC as `RuleBasedRegimeClassifier`,
so nothing about market_regime.py's existing interface changes — this module
only adds a second, alternative implementation of it.

Design carried over from the validated v27_hmm/ research module (2~6 state
sweep on real AAPL/QQQ history — see project memory): 4 fixed states, GaussianHMM
with covariance_type='full', the same 4 features (log_return / volatility /
hl_range / volume_change) with rolling(closed='left') to avoid look-ahead
bias, and adaptive label mapping (never a hard-coded state->label table,
since HMM state indices are arbitrary per training run).

Differences from v27_hmm/ (deliberate, not oversights):
  - v27_hmm/ is a single-run train/test research sandbox with no persistence;
    this module trains once per stock and persists the fitted model (pickle)
    to engine/hmm_models/, since production needs to *decode* new bars
    against a fixed, already-trained model, not retrain from scratch daily.
  - v27_hmm/ operates on the capitalized Open/High/Low/Close/Volume schema;
    kabu's own data pipeline (data/fetcher.py, backtest.py) uses lowercase
    open/high/low/close/volume — the feature engineering here is duplicated
    (not imported from v27_hmm/) and adapted to that schema, following the
    same "duplicate rather than cross-import" discipline already established
    between engine/regime.py and engine/regime_features.py in this codebase.
  - N_STATES is fixed at 4 with named labels [Bear, Correction, Sideways,
    Bull] — the state-count research phase (v27_hmm/) is intentionally over;
    see project memory for why 4 was chosen over the per-asset-optimal
    numbers that research surfaced (AAPL preferred 4, QQQ's Overall score
    preferred 6 with 3 a close/cleaner second — no single N was universally
    best, so this stage locks a single practical choice rather than
    continuing to chase a moving target).

Data source: training uses backtest.py's fetch_kline() / data_cache/ cache,
NOT data/fetcher.py (the live intraday runner's data path, capped at roughly
a year of history by config._LOOKBACK_DAYS — nowhere near enough for a
reliable 4-state fit). This keeps HMM training and decoding fully decoupled
from engine/runner.py's live trading data path, by construction.

NOT wired into engine/runner.py, backtest_portfolio.py, or any trading
decision as of V2.7 Stage 3. The only consumer as of this stage is
engine/hmm_shadow.py's standalone Shadow Mode script.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import config
from data_provider.provider_factory import get_provider
from engine.market_regime import Regime, RegimeClassifier, REGIME_NAMES

_provider = get_provider(config.MARKET)

HMM_N_STATES = 4
HMM_RANDOM_STATE = 42
HMM_N_ITER = 1000
HMM_COVARIANCE_TYPE = "full"
HMM_LOOKBACK_PERIOD = 5
HMM_ATR_PERIOD = 14
HMM_FEATURE_LIST = ["log_return", "volatility", "hl_range", "volume_change"]

# Below this many usable (post rolling-warmup) bars, refuse to train — a
# 4-state GaussianHMM fit on too little history is not a regime model, it's
# noise. v27_hmm/'s validated runs used ~2900-3600 daily bars; this is a
# conservative floor, not the recommended amount.
HMM_MIN_TRAIN_BARS = 500

HMM_VERSION = "hmm-v1"  # bump when the feature set / state count / label
                         # scheme changes, so persisted models and shadow log
                         # rows can be told apart from a future retraining.

HMM_MODEL_DIR = Path(__file__).parent / "hmm_models"

_LABEL_TO_REGIME = {
    "Bear": Regime.HMM_BEAR,
    "Correction": Regime.HMM_CORRECTION,
    "Sideways": Regime.HMM_SIDEWAYS,
    "Bull": Regime.HMM_BULL,
}


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must be indexed by date with lowercase columns open/high/low/close/
    volume (kabu's schema — see module docstring). Same 4 features and
    look-ahead-bias discipline as v27_hmm/features.py::build_features(),
    duplicated here rather than imported.
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["volatility"] = (
        df["log_return"].rolling(window=HMM_LOOKBACK_PERIOD, closed="left").std()
    )
    volume_baseline = df["volume"].rolling(window=HMM_LOOKBACK_PERIOD, closed="left").mean()
    df["volume_change"] = df["volume"] / volume_baseline - 1

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Relative ATR (divided by close), same convention as v27_hmm/main.py's
    # run_pipeline() — comparable across price levels, displayable as a %.
    df["atr"] = true_range.rolling(window=HMM_ATR_PERIOD, closed="left").mean() / df["close"]
    return df


def prepare_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a backtest.fetch_kline() result into a date-indexed frame,
    sorted and de-duplicated, ready for _build_features().

    backtest.fetch_kline()'s OHLCV columns come back as object dtype (boxed
    Python floats from the moomoo SDK, not numpy float64) — np.log/rolling
    silently fail on that dtype, so this must coerce to numeric explicitly,
    same fix v27_hmm/features.py::clean_data() already applies to its own
    (differently-cased) CSV input."""
    df = raw.rename(columns={"time_key": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def _map_states_to_labels(df: pd.DataFrame, state_col: str = "state") -> dict:
    """
    Adaptive label mapping for the fixed 4-state scheme, sorted by each
    state's mean log_return within the training set (same principle as
    v27_hmm/hmm_model.py::map_states_to_labels() — HMM state indices are
    arbitrary per run, never hard-code {0: 'Bull', ...}):

        lowest mean return  -> Bear
        2nd lowest          -> Correction  (high-vol, negative-tilt middle state)
        2nd highest         -> Sideways    (low-vol, calmer middle state)
        highest mean return -> Bull
    """
    stats = df.groupby(state_col)["log_return"].mean().sort_values()
    state_ids = stats.index.tolist()
    if len(state_ids) != HMM_N_STATES:
        raise ValueError(
            f"expected {HMM_N_STATES} distinct decoded states in the training "
            f"set, got {len(state_ids)} — degenerate fit, do not use this model"
        )
    return {
        state_ids[0]: "Bear",
        state_ids[1]: "Correction",
        state_ids[2]: "Sideways",
        state_ids[3]: "Bull",
    }


def _model_path(code: str) -> Path:
    return HMM_MODEL_DIR / f"{code.replace('.', '_')}.pkl"


def _save_model(code: str, payload: dict) -> None:
    HMM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(code)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_stock_hmm(code: str) -> dict | None:
    """Returns the persisted {model, scaler, label_map, ...} payload for
    `code`, or None if it hasn't been trained yet."""
    path = _model_path(code)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def train_stock_hmm(code: str, start: str = "2015-01-01", end: str = None) -> dict:
    """
    Train a 4-state GaussianHMM on the full available history for `code`
    (via backtest.fetch_kline()'s cache — see module docstring) and persist
    it to engine/hmm_models/. Trains on ALL available history (no train/test
    split) — unlike v27_hmm/'s research pipeline, production wants the model
    fit on as much data as possible before decoding "today".

    Raises ValueError if there isn't enough history (HMM_MIN_TRAIN_BARS) or
    the fit degenerates (a state with zero occupancy) — callers (the CLI
    below, engine/hmm_shadow.py) must catch this per-stock rather than let
    one bad ticker abort a batch.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    raw = _provider.get_history(code, start=start, end=end)
    if raw.empty:
        raise ValueError(f"{code}: no data returned for [{start}, {end}]")

    df = prepare_ohlcv(raw)
    df = _build_features(df)
    df = df.dropna(subset=HMM_FEATURE_LIST)

    if len(df) < HMM_MIN_TRAIN_BARS:
        raise ValueError(
            f"{code}: only {len(df)} usable bars after feature warmup, "
            f"need >= {HMM_MIN_TRAIN_BARS} to train a reliable 4-state model"
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[HMM_FEATURE_LIST])

    model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE,
    )
    model.fit(scaled)

    df["state"] = model.predict(scaled)
    label_map = _map_states_to_labels(df, "state")

    payload = {
        "code": code,
        "model": model,
        "scaler": scaler,
        "label_map": label_map,
        "feature_list": HMM_FEATURE_LIST,
        "hmm_version": HMM_VERSION,
        "trained_at": pd.Timestamp.now().isoformat(),
        "train_start": str(df.index[0].date()),
        "train_end": str(df.index[-1].date()),
        "n_samples": len(df),
    }
    _save_model(code, payload)
    return payload


HMM_MIN_DECODE_BARS = 60     # need at least this much trailing history before
                              # the first causal decode is trusted
HMM_DECODE_WINDOW = 756      # ~3 trading years of trailing context per decode


def decode_regime_series_causal(
    payload: dict, ohlcv_df: pd.DataFrame, window: int = HMM_DECODE_WINDOW
) -> pd.DataFrame:
    """
    Walk-forward, no-look-ahead regime decoding — for BACKTESTING/ANALYTICS
    only (V2.7 Stage 4). At each day t, decodes using only a trailing window
    of observations ending at t, never anything after t, and keeps only that
    day's state (and its posterior confidence) from the resulting Viterbi
    path.

    ohlcv_df: date-indexed OHLCV frame already normalized by prepare_ohlcv()
    (raw lowercase open/high/low/close/volume, numeric dtype) — this builds
    the 4-feature set internally, same as train_stock_hmm()/MLRegimeClassifier
    do, so callers never need engine.hmm_regime's private feature-building
    helper directly.

    This is deliberately NOT the same as calling model.predict() once on the
    whole series (which is what MLRegimeClassifier.classify_series() does,
    and what engine/hmm_shadow.py relies on for live use — safe there only
    because Shadow Mode always decodes "history up to today" and discards
    everything but the last row, so "the future" genuinely doesn't exist yet
    when it runs). A backtest/analytics pass that decoded the ENTIRE
    historical series in one batch call would let the Viterbi algorithm
    choose each day's state using the GLOBALLY most likely path across the
    whole series — meaning a day in 2018 could be labeled using information
    from 2026. That is look-ahead bias, and would invalidate any conclusion
    drawn from it (a trading rule OR a "which regime made money" statistic).

    The HMM's own parameters (transition matrix, per-state means/covariances)
    are NOT retrained here — they're the fixed, already-trained model from
    train_stock_hmm(). Only the decoded state path changes per day, using a
    bounded trailing window rather than full expanding history: HMM state
    estimates stabilize within a few hundred bars given this model's observed
    self-transition probabilities (see project memory — 70-95% typical), so
    a ~3-year window loses negligible fidelity vs. decoding on full history,
    while keeping this function's runtime roughly linear instead of quadratic
    in the length of the series.

    Returns a DataFrame indexed like ohlcv_df with columns:
      regime     : label string ("Bear"/"Correction"/"Sideways"/"Bull"), NaN
                   before HMM_MIN_DECODE_BARS of history has accumulated.
      confidence : posterior probability (0-100) of the decoded state within
                   its trailing window (model.predict_proba, forward-backward
                   smoothed over [t-window+1, t] only — never sees day t+1).
      duration   : consecutive days (ending at, and including, that day) the
                   regime label has been unchanged. Computed as a vectorized
                   post-process over the completed `regime` column (not
                   inside the day-by-day loop) — still strictly causal, since
                   "how many days in a row has today's label held" only
                   depends on labels up to and including today, each of
                   which was already decoded without look-ahead.
    """
    model = payload["model"]
    scaler = payload["scaler"]
    label_map = payload["label_map"]
    feature_list = payload["feature_list"]

    feat_df = _build_features(ohlcv_df)
    valid = feat_df.dropna(subset=feature_list)
    if len(valid) < HMM_MIN_DECODE_BARS:
        return pd.DataFrame(
            {"regime": None, "confidence": np.nan, "duration": np.nan}, index=feat_df.index
        )

    scaled_full = scaler.transform(valid[feature_list])
    n = len(valid)
    labels = np.full(n, None, dtype=object)
    confidences = np.full(n, np.nan)

    for i in range(HMM_MIN_DECODE_BARS - 1, n):
        lo = max(0, i - window + 1)
        window_scaled = scaled_full[lo : i + 1]
        state_path = model.predict(window_scaled)
        state_t = state_path[-1]
        posteriors = model.predict_proba(window_scaled)
        labels[i] = label_map[state_t]
        confidences[i] = posteriors[-1, state_t] * 100.0

    out = pd.DataFrame(
        {"regime": labels, "confidence": confidences}, index=valid.index
    ).reindex(feat_df.index)

    change = out["regime"] != out["regime"].shift(1)
    run_id = (change | out["regime"].isna()).cumsum()
    duration = out.groupby(run_id).cumcount() + 1
    out["duration"] = duration.where(out["regime"].notna())
    return out


class MLRegimeClassifier(RegimeClassifier):
    """
    Stage-2 HMM-based implementation of engine/market_regime.py's
    RegimeClassifier interface. Unlike RuleBasedRegimeClassifier (stateless,
    computes thresholds fresh from any df), one instance is bound to a
    single stock's pre-trained model — the HMM parameters are fit offline
    (train_stock_hmm) and only *decoded* here, never re-fit per call.

    Raises FileNotFoundError at construction if no trained model exists for
    `code`, rather than silently falling back to an untrained/garbage model —
    callers must train first (or catch and skip that stock).
    """

    def __init__(self, code: str, model_payload: dict = None):
        self.code = code
        self.payload = model_payload if model_payload is not None else load_stock_hmm(code)
        if self.payload is None:
            raise FileNotFoundError(
                f"No trained HMM model for {code}; call train_stock_hmm('{code}') first"
            )

    def classify_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must be date-indexed with lowercase open/high/low/close/volume
        (kabu's schema — see prepare_ohlcv). Returns a DataFrame matching
        RegimeClassifier's contract (regime, regime_name, adx, atr, atr_pct,
        bb_width, volume_ratio, confidence) so it's a drop-in alternative to
        RuleBasedRegimeClassifier wherever a RegimeClassifier is expected.

        adx and bb_width have no HMM analog and are left NaN (rule-based-only
        facts). atr_pct reuses this module's own relative-ATR feature.
        volume_ratio reuses the volume_change feature (same "how unusual is
        today's volume" intent as RuleBasedRegimeClassifier's volume_ratio,
        computed differently — not numerically comparable across the two
        classifiers, only within each).
        """
        model = self.payload["model"]
        scaler = self.payload["scaler"]
        label_map = self.payload["label_map"]
        feature_list = self.payload["feature_list"]

        work = _build_features(df)
        valid = work.dropna(subset=feature_list)

        out = pd.DataFrame(index=df.index)
        out["regime"] = Regime.UNKNOWN
        out["regime_name"] = REGIME_NAMES[Regime.UNKNOWN]
        out["adx"] = np.nan
        out["atr"] = work["atr"]
        out["atr_pct"] = work["atr"]
        out["bb_width"] = np.nan
        out["volume_ratio"] = work["volume_change"]
        out["confidence"] = 0.0

        if valid.empty:
            return out

        scaled = scaler.transform(valid[feature_list])
        states = model.predict(scaled)                 # Viterbi-decoded state path
        posteriors = model.predict_proba(scaled)        # forward-backward posteriors

        labels = pd.Series(states, index=valid.index).map(label_map)
        regimes = labels.map(_LABEL_TO_REGIME)
        # Confidence = posterior probability of the Viterbi-decoded state at
        # each bar (0-100) — a genuine model-derived confidence, not just a
        # hard 0/1 label, consistent with RegimeClassifier's confidence contract.
        confidence = posteriors[np.arange(len(states)), states] * 100

        out.loc[valid.index, "regime"] = regimes.values
        out.loc[valid.index, "regime_name"] = labels.values
        out.loc[valid.index, "confidence"] = confidence
        return out


def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train per-stock HMM regime models (V2.7 Stage 3, Step 1)"
    )
    parser.add_argument("--codes", nargs="+", required=True, help="e.g. US.AAPL US.QQQ")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    for code in args.codes:
        try:
            payload = train_stock_hmm(code, start=args.start)
            print(
                f"[ok] {code}: trained on {payload['n_samples']} bars "
                f"({payload['train_start']} -> {payload['train_end']}), "
                f"label_map={payload['label_map']}"
            )
        except Exception as e:
            print(f"[error] {code}: {e!r}")


if __name__ == "__main__":
    _cli_main()
