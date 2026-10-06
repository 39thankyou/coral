"""Train Elasticity through the author INR and regression entrypoints."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_adaptor.author_pipeline import main


if __name__ == "__main__":
    main("elasticity")
