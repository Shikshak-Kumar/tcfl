"""
Backward-compatible entry: use **run_all_research.py** (one command does everything).

This module delegates to run_all_research.main().
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_all_research import main  # noqa: E402

if __name__ == "__main__":
    main()
