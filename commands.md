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
./describe data/dataset_train.csv
```
Affiche count / mean / std / min / 25% / 50% / 75% / max des 13 matières,
avec nos propres fonctions statistiques.

### V.2 — Data Visualization

```bash
./histogram data/dataset_train.csv
```
Classement d'homogénéité en console, puis grille des 13 matières.
**Réponse : Arithmancy** (0.0292, contre 0.8876 pour la 3e).

```bash
./scatter_plot data/dataset_train.csv
```
Classement des corrélations, puis nuage du couple le plus similaire.
**Réponse : Astronomy / Defense Against the Dark Arts**, r = -1.0000.

```bash
./pair_plot data/dataset_train.csv
```
Matrice 13×13 de toutes les paires. Sert à choisir les features.

### V.3 — Logistic Regression

```bash
./logreg_train data/dataset_train.csv
```
Entraîne les 4 classifieurs one-vs-all par **batch** gradient descent
(800 itérations), écrit `data/theta.csv`, affiche les learning curves.

```bash
./logreg_predict data/dataset_test.csv data/theta.csv
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
./logreg_predict data/dataset_train.csv data/theta.csv
```
Rejoue la prédiction, cette fois sur le jeu d'entraînement.

```bash
python -m src.evaluation.confusion_matrix data/dataset_train.csv data/houses.csv
```
Compare prédictions et vérité : montre où le modèle se trompe, maison par maison.

```bash
python -m src.evaluation.personal_probabilities data/dataset_train.csv data/theta.csv
```
Affiche les probabilités des 4 maisons pour les élèves ambigus.

```bash
./logreg_predict data/dataset_test.csv data/theta.csv
```
Remet `houses.csv` dans son état normal (prédictions sur le jeu de test).

---

## 3. Bonus — Optimiseurs alternatifs

> Les deux descentes sont dans `src/bonus/`. Un seul programme pour les
> deux, l'option choisit l'algorithme et les fichiers de sortie.
> **Aucun fichier du mandatory n'est modifié.**

### Stochastic gradient descent

```bash
./logreg_train_bonus --sgd data/dataset_train.csv
```
Corrige theta **après chaque élève** : 1600 corrections par epoch au lieu
d'une. 20 epochs suffisent, contre 800 itérations pour le batch.
Écrit `data/theta_sgd.csv`.

```bash
./logreg_predict_bonus --sgd data/dataset_test.csv data/theta_sgd.csv
```
Prédit avec ces poids → `data/houses_SGD.csv`.

### Mini-batch gradient descent

```bash
./logreg_train_bonus --mbgd data/dataset_train.csv
```
Le compromis : corrige après chaque **paquet de 32 élèves**, soit 50
corrections par epoch. Écrit `data/theta_mbgd.csv`.

```bash
./logreg_predict_bonus --mbgd data/dataset_test.csv data/theta_mbgd.csv
```
Prédit avec ces poids → `data/houses_MBGD.csv`.

### Comparer les trois

```bash
diff data/houses.csv data/houses_SGD.csv
```
```bash
diff data/houses.csv data/houses_MBGD.csv
```
**Aucune sortie = prédictions identiques** à celles du batch. C'est la
validation des bonus.

```bash
diff data/theta.csv data/theta_sgd.csv
```
Montre que les poids, eux, **diffèrent** — donc que la comparaison
précédente a bien du sens.

---

## Les trois descentes en un tableau

| | Élèves par correction | Corrections / epoch |
|---|---|---|
| Batch (mandatory) | 1600 | 1 |
| Mini-batch (bonus) | 32 | 50 |
| Stochastique (bonus) | 1 | 1600 |

Même modèle, même fonction de coût, même gradient — seule change la
quantité de données utilisée avant chaque correction de theta.
