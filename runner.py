"""
runner.py — compatibility shim.  The canonical entry point is now main.py.

    python main.py --auto --confirmed --interval 300

This file is kept so existing Task Scheduler jobs and scripts that reference
runner.py continue to work without changes.
"""
import sys
import warnings

warnings.warn(
    "runner.py is deprecated — use main.py instead.  "
    "Forwarding to main.py now.",
    DeprecationWarning,
    stacklevel=1,
)

from main import cli_main   # noqa: E402

if __name__ == "__main__":
    cli_main()
