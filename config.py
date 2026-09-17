from pathlib import Path


ROOT = Path(__file__).parent
DATA: Path = ROOT / "data"
THETA_FILE: Path = DATA / "theta.csv"
RESULT_FILE: Path = DATA / "houses.csv"

LEARNING_RATE = 0.03
N_ITERATION = 500

RELEVANT_FEATURES = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        "Charms"
]
