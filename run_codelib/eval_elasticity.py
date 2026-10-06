"""Evaluate Elasticity checkpoints and save per-case and INR reconstruction metrics."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_codelib.eval_airfoil import evaluate, parse_args


if __name__ == "__main__":
    evaluate(parse_args("elasticity"))
