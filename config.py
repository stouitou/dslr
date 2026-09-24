from pathlib import Path


ROOT = Path(__file__).parent
DATA: Path = ROOT / "data"
THETA_FILE: Path = DATA / "theta.csv"
RESULT_FILE: Path = DATA / "houses.csv"

LEARNING_RATE = 0.1
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
        "Ancient Runes"
]

# --- Bonus : stochastic gradient descent ---------------------------------
# Fichier de poids distinct de THETA_FILE, pour que le bonus n'écrase
# jamais les poids produits par la partie obligatoire.
THETA_SGD_FILE: Path = DATA / "theta_sgd.csv"
# Prédictions du bonus, séparées de RESULT_FILE pour pouvoir comparer les
# deux sorties sans que l'une écrase l'autre.
RESULT_SGD_FILE: Path = DATA / "houses_SGD.csv"
# Pas plus petit que LEARNING_RATE : le SGD corrige theta à chaque élève, donc
# ~1600 fois par passage au lieu d'une seule.
SGD_LEARNING_RATE = 0.01
# Nombre de passages complets sur le dataset (et non d'itérations).
SGD_N_EPOCHS = 20
