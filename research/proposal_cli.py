"""
research/proposal_cli.py — human CLI for the V3.6 Validation Gate
(research/proposal_store.py).

This is the ONLY module allowed to call ProposalStore.approve() (a
structural test enforces it). `approve` additionally refuses to run unless
stdin is an interactive terminal and the operator re-types the proposal id
— so a script, scheduled task or AI agent driving a shell cannot approve a
proposal. Nothing here writes config.py; APPROVED only records sign-off.

Usage (python -m research.proposal_cli ...):
    list [--status S]
    show ID [--version N]
    create --title T --by NAME --set NAME=VALUE [--set ...] [--pattern PID] [--rationale R]
    revise ID --by NAME --set NAME=VALUE [--set ...] [--rationale R]
    backtest ID --version N --result FILE.json --by NAME
    oos ID --version N --result FILE.json --by NAME
    paper-start ID --version N --by NAME
    paper-result ID --version N --result FILE.json --by NAME
    gate ID --version N --by NAME
    reject ID --version N --reason R --by NAME
    approve ID --version N --by NAME            (interactive only)
    [--db PATH] before the subcommand points at a non-default DB.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from research.proposal_store import ProposalError, ProposalStore


def _parse_set(items) -> dict:
    changes = {}
    for item in items or []:
        if "=" not in item:
            raise ProposalError(f"--set expects NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        changes[name.strip()] = value
    return changes


def _load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_row(row: dict) -> None:
    for k, v in row.items():
        if k.endswith("_json") and v:
            v = json.dumps(json.loads(v), ensure_ascii=False, indent=2)
        print(f"{k:22s}: {v}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.proposal_cli")
    p.add_argument("--db", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list"); s.add_argument("--status")
    s = sub.add_parser("show"); s.add_argument("id"); s.add_argument("--version", type=int)

    s = sub.add_parser("create")
    s.add_argument("--title", required=True); s.add_argument("--by", required=True)
    s.add_argument("--set", action="append", required=True)
    s.add_argument("--pattern"); s.add_argument("--rationale", default="")

    s = sub.add_parser("revise"); s.add_argument("id")
    s.add_argument("--by", required=True); s.add_argument("--set", action="append", required=True)
    s.add_argument("--rationale", default="")

    for name in ("backtest", "oos", "paper-result"):
        s = sub.add_parser(name); s.add_argument("id")
        s.add_argument("--version", type=int, required=True)
        s.add_argument("--result", required=True); s.add_argument("--by", required=True)
    for name in ("paper-start", "gate", "approve"):
        s = sub.add_parser(name); s.add_argument("id")
        s.add_argument("--version", type=int, required=True); s.add_argument("--by", required=True)
    s = sub.add_parser("reject"); s.add_argument("id")
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--reason", required=True); s.add_argument("--by", required=True)
    return p


def _approve(store: ProposalStore, args, input_fn: Callable[[str], str],
             isatty: Callable[[], bool]) -> dict:
    if not isatty():
        raise ProposalError("approve must be run from an interactive terminal")
    row = store.get(args.id, args.version)
    _print_row(row)
    typed = input_fn(f"\nType the proposal id to approve {args.id} v{args.version} "
                     f"as {args.by!r}: ").strip()
    if typed != args.id:
        raise ProposalError("confirmation did not match — not approved")
    return store.approve(args.id, args.version, approved_by=args.by)


def main(argv=None, input_fn: Callable[[str], str] = input,
         isatty: Optional[Callable[[], bool]] = None) -> int:
    args = _build_parser().parse_args(argv)
    isatty = isatty or (lambda: sys.stdin is not None and sys.stdin.isatty())
    store = ProposalStore(db_path=Path(args.db) if args.db else None)
    try:
        if args.cmd == "list":
            print(store.list(args.status).to_string(index=False))
            return 0
        if args.cmd == "show":
            _print_row(store.get(args.id, args.version))
            print(store.events(args.id).to_string(index=False))
            return 0
        if args.cmd == "create":
            row = store.create(args.title, _parse_set(args.set), args.by,
                               rationale=args.rationale, source_pattern_id=args.pattern)
        elif args.cmd == "revise":
            row = store.revise(args.id, _parse_set(args.set), args.by, rationale=args.rationale)
        elif args.cmd == "backtest":
            row = store.record_backtest(args.id, args.version, _load_result(args.result), args.by)
        elif args.cmd == "oos":
            row = store.record_oos(args.id, args.version, _load_result(args.result), args.by)
        elif args.cmd == "paper-start":
            row = store.start_paper(args.id, args.version, args.by)
        elif args.cmd == "paper-result":
            row = store.record_paper_result(args.id, args.version,
                                            _load_result(args.result), args.by)
        elif args.cmd == "gate":
            row = store.pass_gate(args.id, args.version, args.by)
        elif args.cmd == "reject":
            row = store.reject(args.id, args.version, args.reason, args.by)
        elif args.cmd == "approve":
            row = _approve(store, args, input_fn, isatty)
        print(f"{row['proposal_id']} v{row['version']} -> {row['status']}")
        return 0
    except ProposalError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
