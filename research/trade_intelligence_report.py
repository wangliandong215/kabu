"""
research/trade_intelligence_report.py — V3.6-A report builder/writer.

Writes both markdown and JSON, timestamped-by-run-id plus an overwritten
"_latest" pair, to config.TRADE_INTELLIGENCE_REPORT_DIR. The narrative call
is wrapped exactly like risk/llm_advisor.py::LLMRiskAdvisor.analyze() — any
exception is caught and replaced with a static fallback string, never
blocks report generation, never affects the `patterns` section.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from research.trade_intelligence_patterns import PatternCandidate
from research.trade_intelligence_narrative import NarrativeProvider

_DISCLAIMER = (
    "How to read this report: every finding below is either an "
    "OBSERVATION (a pattern noticed in the data, below the sample-size bar "
    "for anything stronger) or a CANDIDATE_PATTERN (cleared a higher "
    "sample-size + effect-size/precision bar). Neither state is a proven "
    "cause, a validated trading rule, or an instruction to change any "
    "production strategy, position sizing, exit, or risk logic. Promotion "
    "past CANDIDATE_PATTERN requires Backtest + Out-of-Sample validation + "
    "Paper Trading + human approval — none of which this report performs."
)


def _narrative_text(provider: NarrativeProvider, context: Dict[str, Any]) -> str:
    try:
        return provider.generate(context).summary
    except Exception as exc:
        try:
            import notify.alert as alert
            alert.log(f"trade_intelligence_report: narrative generation failed "
                      f"({exc.__class__.__name__}: {exc}) — using fallback text")
        except Exception:
            pass
        return "AI narrative unavailable this run (generation failed)."


def build_report(patterns: List[PatternCandidate], run_summary: Dict[str, Any],
                  provider: NarrativeProvider) -> Dict[str, Any]:
    analysis_types = sorted({p.analysis_type for p in patterns})
    context = {
        "n_trades_analyzed": run_summary.get("n_trades_analyzed", 0),
        "n_candidate_patterns": sum(1 for p in patterns if p.state == "CANDIDATE_PATTERN"),
        "n_observations": sum(1 for p in patterns if p.state == "OBSERVATION"),
        "analysis_types": analysis_types,
    }
    narrative = _narrative_text(provider, context)
    return {
        "run_summary": run_summary,
        "disclaimer": _DISCLAIMER,
        "narrative": narrative,
        "patterns": [
            {
                "analysis_type": p.analysis_type, "pattern_key": p.pattern_key,
                "state": p.state, "description": p.description,
                "metric_name": p.metric_name, "metric_value": p.metric_value,
                "comparison_metric_value": p.comparison_metric_value,
                "effect_size": p.effect_size, "n_sample": p.n_sample,
                "n_baseline": p.n_baseline, "precision": p.precision,
                "recall": p.recall, "false_positive_count": p.false_positive_count,
                "slice_definition": p.slice_definition,
            }
            for p in patterns
        ],
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    lines = ["# V3.6-A Trade Intelligence Report", ""]
    rs = payload["run_summary"]
    lines.append(f"- run_id: {rs.get('run_id')}")
    lines.append(f"- generated_at: {rs.get('finished_at')}")
    lines.append(f"- trades analyzed: {rs.get('n_trades_analyzed')}")
    lines.append("")
    lines.append(payload["disclaimer"])
    lines.append("")
    lines.append("## AI-generated narrative — advisory summary only, not itself a finding")
    lines.append("")
    lines.append(payload["narrative"])
    lines.append("")

    by_type: Dict[str, List[dict]] = {}
    for p in payload["patterns"]:
        by_type.setdefault(p["analysis_type"], []).append(p)

    for analysis_type in sorted(by_type):
        lines.append(f"## {analysis_type}")
        lines.append("")
        for p in by_type[analysis_type]:
            lines.append(f"### [{p['state']}] {p['pattern_key']}")
            lines.append(p["description"])
            detail = (f"- metric: {p['metric_name']}={p['metric_value']}, "
                      f"baseline={p['comparison_metric_value']}, "
                      f"effect_size={p['effect_size']}, n={p['n_sample']}")
            if p["precision"] is not None:
                detail += f", precision={p['precision']}, recall={p['recall']}"
            lines.append(detail)
            lines.append("")

    if not by_type:
        lines.append("_No patterns cleared the minimum sample-size bar this run._")
        lines.append("")
    return "\n".join(lines)


def write_report(payload: Dict[str, Any], report_dir: Optional[Path] = None) -> Dict[str, Path]:
    out_dir = Path(report_dir or config.TRADE_INTELLIGENCE_REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_summary"].get("run_id") or datetime.now().strftime("%Y%m%dT%H%M%S")

    md_text = _render_markdown(payload)
    json_text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)

    paths: Dict[str, Path] = {}
    for stem, key_prefix in ((f"trade_intelligence_{run_id}", "run"),
                              ("trade_intelligence_latest", "latest")):
        md_path = out_dir / f"{stem}.md"
        json_path = out_dir / f"{stem}.json"
        md_path.write_text(md_text, encoding="utf-8")
        json_path.write_text(json_text, encoding="utf-8")
        paths[f"{key_prefix}_md"] = md_path
        paths[f"{key_prefix}_json"] = json_path
    return paths
