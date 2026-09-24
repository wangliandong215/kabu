"""
research/trade_intelligence_quality.py — V3.6 Trade Intelligence data
quality report (research-only).

build_quality_report(population) takes research.trade_intelligence_data.
load_population()'s frame (every trade, classified + inclusion-marked) and
reports:
  * population — how every trade was classified and why it was included in
    or excluded from the analyzed set (nothing is dropped silently);
  * field fill rates — per core field, how many rows actually carry a
    recorded value, over three denominators: all trades, closed non-phantom
    trades, and the analyzed set. Exit-side fields only count closed trades.

A missing value is reported as missing. Nothing here estimates, imputes or
back-fills anything — the point is to show how complete the recorded
history really is.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config

# (label, column, scope). scope "entry" = meaningful for every trade,
# "exit" = only for closed trades (open trades are not counted as missing).
CORE_FIELDS = [
    ("entry_price", "entry_price", "entry"),
    ("entry strategy", "strategy_name", "entry"),
    ("entry signal strength (snapshot)", "rs_signal_strength", "entry"),
    ("entry rank", "entry_rank", "entry"),
    ("ATR at entry", "atr_entry", "entry"),
    ("entry regime (HMM)", "entry_regime_label", "entry"),
    ("entry regime features (ADX etc.)", "entry_adx", "entry"),
    ("QQQ state (prior recorded day)", "qqq_prev_close", "entry"),
    ("SPY state (prior recorded day)", "spy_prev_close", "entry"),
    ("VIX", "mc_vix_close", "entry"),
    ("CNN Fear & Greed", "mc_cnn_fear_greed", "entry"),
    ("AAII", "mc_aaii_spread", "entry"),
    ("Put/Call ratio", "mc_put_call_ratio", "entry"),
    ("NAAIM", "mc_naaim_exposure", "entry"),
    ("position_pct", "position_pct", "entry"),
    ("cash_before", "cash_before", "entry"),
    ("equity_before", "equity_before", "entry"),
    ("portfolio exposure (risk log)", "entry_exposure_pct", "entry"),
    ("risk tier (risk log)", "entry_risk_tier", "entry"),
    ("confidence score", "confidence_score", "entry"),
    ("entry quality", "distance_from_20d_high", "entry"),
    ("account_env (metadata)", "account_env", "entry"),
    ("parameter_hash", "parameter_hash", "entry"),
    ("git_commit", "git_commit", "entry"),
    ("raw execution column", "execution", "entry"),
    ("MFE", "mfe", "exit"),
    ("MAE", "mae", "exit"),
    ("holding duration", "holding_hours", "exit"),
    ("daily trajectory (trade_daily)", "daily_n_rows", "entry"),
    ("regime drift", "regime_drifted", "exit"),
    ("exit_price", "exit_price", "exit"),
    ("exit reason code", "exit_reason_code", "exit"),
    ("P/L", "pnl", "exit"),
    ("exit category", "exit_category", "exit"),
    ("MFE capture", "mfe_capture", "exit"),
    ("forward return 5d (outcome)", "outcome_fwd_return_5d", "entry"),
    ("commission", "commission", "exit"),
    ("slippage", "slippage", "exit"),
]

POPULATIONS = ("all", "closed_non_phantom", "analyzed")


def _filled(series: pd.Series) -> int:
    s = series.dropna()
    if s.dtype == object:
        s = s[s.astype(str).str.strip() != ""]
    return int(len(s))


def _subsets(pop: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if pop.empty:
        return {k: pop for k in POPULATIONS}
    closed = pop[(~pop["is_phantom"].astype(bool)) & pop["is_closed"].astype(bool)]
    analyzed = pop[pop["inclusion_status"] == "INCLUDED"] if "inclusion_status" in pop else pop.iloc[0:0]
    return {"all": pop, "closed_non_phantom": closed, "analyzed": analyzed}


def population_summary(pop: pd.DataFrame) -> Dict[str, Any]:
    if pop.empty:
        return {"total": 0, "by_execution_class": {}, "phantom": 0, "open": 0,
                "closed_non_phantom": 0, "analyzed": 0, "inclusion": {},
                "pre_fill_confirmation_fix": 0, "account_env": {},
                "execution_class_reasons": {}}
    counts = lambda s: {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}
    return {
        "total": int(len(pop)),
        "by_execution_class": counts(pop["execution_class"]),
        "raw_execution": counts(pop["execution"].fillna("NULL")),
        "phantom": int(pop["is_phantom"].sum()),
        "open": int((~pop["is_closed"].astype(bool) & ~pop["is_phantom"].astype(bool)).sum()),
        "closed_non_phantom": int(len(_subsets(pop)["closed_non_phantom"])),
        "analyzed": int((pop["inclusion_status"] == "INCLUDED").sum()),
        "inclusion": counts(pop["inclusion_reason"]),
        "inclusion_by_class": {
            f"{c}/{r}": int(n) for (c, r), n in
            pop.groupby(["execution_class", "inclusion_reason"]).size().items()},
        "account_env": counts(pop["account_env"].fillna("NULL (no run_id)")),
        "pre_fill_confirmation_fix": int(pop["pre_fill_confirmation_fix"].sum()),
        "execution_class_reasons": {
            str(c): str(r) for c, r in
            pop.drop_duplicates("execution_class")[["execution_class",
                                                    "execution_class_reason"]].values},
    }


def field_fill_rates(pop: pd.DataFrame) -> List[Dict[str, Any]]:
    subsets = _subsets(pop)
    rows = []
    for label, col, scope in CORE_FIELDS:
        rec = {"field": label, "column": col, "scope": scope}
        for name, df in subsets.items():
            base = df
            if scope == "exit" and not df.empty:
                base = df[df["is_closed"].astype(bool)]
            n = int(len(base))
            k = _filled(base[col]) if (col in base.columns and n) else 0
            rec[name] = {"filled": k, "total": n}
        rows.append(rec)
    return rows


def build_quality_report(pop: pd.DataFrame, run_id: Optional[str] = None,
                         selection: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "selection": selection or {},
        "population": population_summary(pop),
        "fields": field_fill_rates(pop),
        "note": ("Counts are of values actually recorded in trade_history.db / the "
                 "risk_state log. Missing values are NULL/unavailable — nothing is "
                 "estimated or back-filled."),
    }


def _frac(d: dict) -> str:
    return f"{d['filled']}/{d['total']}" if d["total"] else "-"


def render_markdown(report: Dict[str, Any]) -> str:
    p = report["population"]
    lines = ["# Trade Intelligence Data Quality Report", "",
             f"- run_id: {report.get('run_id')}",
             f"- generated_at: {report.get('generated_at')}",
             f"- selection: {json.dumps(report.get('selection', {}), ensure_ascii=False)}",
             "", "## Population", "",
             f"- total trades: {p.get('total', 0)}",
             f"- execution class: {p.get('by_execution_class', {})}",
             f"- raw execution column: {p.get('raw_execution', {})}",
             f"- phantom: {p.get('phantom', 0)}",
             f"- open (non-phantom): {p.get('open', 0)}",
             f"- closed non-phantom: {p.get('closed_non_phantom', 0)}",
             f"- **analyzed: {p.get('analyzed', 0)}**",
             f"- inclusion reasons: {p.get('inclusion', {})}",
             f"- by class/reason: {p.get('inclusion_by_class', {})}",
             f"- account_env (metadata): {p.get('account_env', {})}",
             f"- flagged pre fill-confirmation fix (2026-08-12): "
             f"{p.get('pre_fill_confirmation_fix', 0)}", ""]
    for cls, reason in p.get("execution_class_reasons", {}).items():
        lines.append(f"  - {cls}: {reason}")
    lines += ["", "## Field fill rates", "",
              "| field | scope | all | closed non-phantom | analyzed |",
              "|---|---|---|---|---|"]
    for f in report["fields"]:
        lines.append(f"| {f['field']} | {f['scope']} | {_frac(f['all'])} | "
                     f"{_frac(f['closed_non_phantom'])} | {_frac(f['analyzed'])} |")
    lines += ["", report["note"], ""]
    return "\n".join(lines)


def write_quality_report(report: Dict[str, Any],
                         report_dir: Optional[Path] = None) -> Dict[str, Path]:
    out_dir = Path(report_dir or config.TRADE_INTELLIGENCE_REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = report.get("run_id") or datetime.now().strftime("%Y%m%dT%H%M%S")
    md_text = render_markdown(report)
    json_text = json.dumps(report, indent=2, default=str, ensure_ascii=False)
    paths: Dict[str, Path] = {}
    for stem, key in ((f"data_quality_{run_id}", "run"), ("data_quality_latest", "latest")):
        (out_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")
        (out_dir / f"{stem}.json").write_text(json_text, encoding="utf-8")
        paths[f"{key}_md"] = out_dir / f"{stem}.md"
        paths[f"{key}_json"] = out_dir / f"{stem}.json"
    return paths
