from pathlib import Path


ROOT = Path(__file__).parent
DATA: Path = ROOT / "data"
THETA_FILE: Path = DATA / "theta.csv"
RESULT_FILE: Path = DATA / "houses.csv"

N_EPOCHS = 0.1
N_ITERATION = 800

COLORMAP = {
        "Gryffindor": "red",
        "Hufflepuff": "yellow",
        "Ravenclaw": "blue",
        "Slytherin": "green"
}
RELEVANT_FEATURES = [
        "Astronomy",
        "Herbology",
        "Defense Against the Dark Arts",
        "Ancient Runes",
        "Charms"
]
