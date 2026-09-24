# Commandes pour la soutenance — DSLR

---

## 0. Préparation

```bash
python3 -m venv .venv
```
Crée l'environnement Python isolé. **Une seule fois.**

```bash
source .venv/bin/activate
```
Active l'environnement. **À chaque nouveau terminal.** Le prompt affiche `(.venv)`.

```bash
pip install -r requirements.txt
```
Installe pandas, tabulate et matplotlib (numpy arrive en dépendance).

---

## 1. Partie obligatoire

### V.1 — Data Analysis

```bash
./scripts/describe data/dataset_train.csv
```
Affiche count / mean / std / min / 25% / 50% / 75% / max des 13 matières,
avec nos propres fonctions statistiques.
Affiche var / range en plus pour les bonus

### V.2 — Data Visualization

```bash
./scripts/histogram data/dataset_train.csv
```
Classement d'homogénéité en console, puis grille des 13 matières.
**Réponse : Arithmancy** (0.0292, contre 0.8876 pour la 3e).

```bash
./scripts/scatter_plot data/dataset_train.csv
```
Classement des corrélations, puis nuage du couple le plus similaire.
**Réponse : Astronomy / Defense Against the Dark Arts**, r = -1.0000.

```bash
./scripts/pair_plot data/dataset_train.csv
```
Matrice 13×13 de toutes les paires. Sert à choisir les features.
**Réponse : Astronomy / Herbology / Ancient Runes**, nuages separes

### V.3 — Logistic Regression

```bash
./scripts/logreg_train data/dataset_train.csv
```
Entraîne les 4 classifieurs one-vs-all par **batch** gradient descent
(800 itérations), écrit `data/theta.csv`, affiche les learning curves.

```bash
./scripts/logreg_predict data/dataset_test.csv data/theta.csv
```
Prédit les maisons des 400 élèves du jeu de test → `data/houses.csv`.

```bash
head -5 data/houses.csv && wc -l data/houses.csv
```
Vérifie le format imposé (`Index,Hogwarts House`) et le compte (401 lignes).

---

## 2. Bonus — Évaluation du modèle

> ⚠️ La matrice de confusion a besoin de prédictions **sur un dataset
> dont on connaît les vraies maisons** : il faut donc prédire sur
> `dataset_train.csv` d'abord.

```bash
./scripts/logreg_predict data/dataset_train.csv data/theta.csv
```
Rejoue la prédiction, cette fois sur le jeu d'entraînement.

```bash
./scripts/confusion_matrix data/dataset_train.csv data/houses.csv
```
Compare prédictions et vérité : montre où le modèle se trompe, maison par maison.

```bash
./scripts/personal_probabilities data/dataset_train.csv data/theta.csv
```
Affiche les probabilités des 4 maisons pour les élèves ambigus.

```bash
./scripts/logreg_predict data/dataset_test.csv data/theta.csv
```
Remet `houses.csv` dans son état normal (prédictions sur le jeu de test).

---

## 3. Bonus — Stochastic Gradient Descent

```bash
./scripts/logreg_train_sgd data/dataset_train.csv
```
Même modèle, descente **stochastique** : correction après chaque élève,
20 epochs au lieu de 800 itérations. Écrit `data/theta_sgd.csv`.

```bash
./scripts/logreg_predict_sgd data/dataset_test.csv data/theta_sgd.csv
```
Prédit avec les poids du SGD → `data/houses_SGD.csv` (fichier distinct,
pour ne pas écraser la sortie du mandatory).

```bash
diff data/houses.csv data/houses_SGD.csv
```
**Aucune sortie = les 400 prédictions sont identiques** à celles du batch.
C'est la validation du bonus.

```bash
diff data/theta.csv data/theta_sgd.csv
```
Montre que les poids, eux, **diffèrent** — donc que la comparaison
précédente a bien du sens.

---

