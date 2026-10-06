"""Run the cylinder_flow shared-INR pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_codelib.pipeline import main
if __name__ == "__main__":
    main("cylinder_flow", default_stage="all")
