"""Train Elasticity INR and regression using the shared-INR pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_codelib.pipeline import main


if __name__ == "__main__":
    main("elasticity", default_stage="all")
