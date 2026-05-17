from __future__ import annotations

import sys
from pathlib import Path

_tools = Path(__file__).resolve().parent.parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

from cvi_batch_analysis.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
