"""Evaluate saved shallow_water shared-INR checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_codelib.extended import evaluate, parse_args
if __name__ == "__main__":
    evaluate(parse_args("shallow_water"))
