from pathlib import Path


ROOT = Path(__file__).parent
DATA: Path = ROOT / "data"
THETA_FILE: Path = DATA / "theta.txt"

LEARNING_RATE = 0.03
N_ITERATION = 500
