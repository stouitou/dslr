# Régression logistique — schéma d'ensemble

## En une phrase

> Une **régression logistique**, c'est une régression linéaire (`θᵀx`)
> dont le résultat passe dans la fonction logistique (`g`) pour devenir
> une probabilité. Le **one-vs-all** consiste à en entraîner une par
> maison, et à retenir celle qui répond le plus fort.

⚠️ Le mot « régression » surprend pour un classifieur. Le modèle fait bien
une régression — **sur une probabilité**. La classification n'arrive
qu'après, quand *toi* tu compares les 4 probabilités.

---

## Les deux phases

```
PHASE 1 — ENTRAÎNEMENT          PHASE 2 — PRÉDICTION
   (une seule fois)                (à chaque usage)

 dataset_train.csv                dataset_test.csv
 (maisons connues)                (maisons inconnues)
        │                                │
        ▼                                ▼
   on CHERCHE theta   ──→ theta.csv ──→  on APPLIQUE theta
                                         │
                                         ▼
                                    houses.csv
```

`theta.csv` est le seul lien entre les deux : **résultat** de
l'apprentissage, **matière première** de la prédiction.

---

# PHASE 1 — Comment on trouve les theta

## Le principe : un tâtonnement guidé

On ne calcule pas les theta par une formule directe. On les **cherche**,
par essais successifs :

```
  partir de zéro
        │
        ▼
  ┌──▶ prédire avec les theta actuels        (a)
  │         │
  │         ▼
  │    mesurer l'erreur totale               (b)
  │         │
  │         ▼
  │    corriger les theta                    (c)
  └─────────┘
       800 fois
```

Au premier tour les theta valent 0, donc toutes les probabilités valent
0,5 : le modèle répond au hasard. Après 800 tours, il a convergé.

## Le déroulement complet

```
 1600 élèves, 5 notes chacun, vraie maison connue
        │
        │   ┌──────────── on répète 4 fois, une par maison ───────────┐
        │   │                                                         │
        └──▶│  1. y = 1 si l'élève est de CETTE maison, 0 sinon       │
            │     (one-vs-all : les 3 autres maisons deviennent "0")  │
            │                                                         │
            │  2. theta = [0, 0, 0, 0, 0, 0]   ← on part de zéro      │
            │                                                         │
            │  3. répéter 800 fois : (a) prédire                      │
            │                        (b) mesurer                      │
            │                        (c) corriger                     │
            │                                                         │
            │  4. theta est trouvé pour cette maison                  │
            └─────────────────────────────────────────────────────────┘
                              │
                              ▼
                   4 jeux de 6 poids  →  data/theta.csv
```

**Ce qui change d'une maison à l'autre** : uniquement `y`, les étiquettes.
Les notes sont rigoureusement les mêmes.
**Ce qui change d'un tour à l'autre** : uniquement `theta`.

---

## (a) Prédire

Le calcul de la phase 2 (voir plus bas), avec les theta **du moment** —
encore faux au début. On obtient une probabilité par élève.

## (b) Mesurer l'erreur — la fonction de coût `J(θ)`

Un seul nombre : à quel point le modèle se trompe sur les 1600 élèves.
C'est cette valeur qu'on voit descendre sur les *learning curves*.

**La question posée pour chaque élève** :
> quelle probabilité le modèle a-t-il donnée à la **bonne** réponse ?

```
 l'élève EST de cette maison (y=1)   →  on regarde  h
 l'élève N'EST PAS de cette maison   →  on regarde  1 - h
```

La formule du sujet écrit ce `if/else` en multipliant chaque branche par
`y` ou `(1-y)` : celle dont on ne veut pas est éteinte par un zéro.

**Puis on prend le logarithme**, parce qu'il punit très sévèrement les
erreurs commises avec confiance :

| Proba donnée à la bonne réponse | Coût |
|---|---|
| 0,99 | **0,01**  ← quasi parfait |
| 0,50 | 0,693 ← le hasard |
| 0,01 | **4,61**  ← catastrophique |

`log(1) = 0` : une prédiction parfaite ne coûte rien.
`log(0) = -∞` : une prédiction totalement fausse coûte l'infini.

⚠️ **Pourquoi pas l'erreur quadratique** (comme en régression linéaire) ?
Deux raisons : elle est bornée, donc elle ne distingue pas « faux » de
« absurdement faux » ; et surtout, avec une sigmoïde à l'intérieur, elle
produit une surface **bosselée** où la descente peut rester coincée. Le
log-loss donne une surface en **cuvette** : un seul minimum.

## (c) Corriger — le gradient

`J(θ)` est une surface. On cherche le point le plus bas. La **dérivée
partielle** selon un poids répond à :

> « si j'augmente ce poids d'un poil, l'erreur monte-t-elle ou descend-elle ? »

Il y a **6 poids, donc 6 dérivées** (une par poids). Ensemble, elles
forment le *gradient*.

| Dérivée | Ce qu'elle dit | Ce qu'on fait |
|---|---|---|
| **positive** | augmenter ce poids fait **monter** l'erreur | on le **diminue** |
| **négative** | augmenter ce poids fait **baisser** l'erreur | on **l'augmente** |

→ on va **toujours à l'opposé du signe de la pente**.

```
        \                     /
         \                   /       à gauche : pente négative → augmenter θ
          \_______  _______/         à droite : pente positive → diminuer θ
                  \/
              le minimum
```

**Chaque poids reçoit sa propre correction**, calculée à partir de deux
facteurs :

```
 (prédiction - vérité)  ×  la note de l'élève dans cette matière
  └── de combien on ──┘     └── la responsabilité de ce poids ──┘
      se trompe                  dans l'erreur
```

Si la note est nulle, ce poids n'a rien fait pour cet élève : aucune
correction. Si elle est énorme, il a beaucoup contribué : grosse correction.

## Le pas : le *learning rate*

```
 nouveau θ = θ - (learning rate) × (sa dérivée)
```

* **trop grand** → on saute par-dessus le minimum, la courbe oscille
* **trop petit** → on converge, mais très lentement

---

# PHASE 2 — Prédiction

## Étape A : calculer θᵀx (une somme pondérée)

Pour **un** élève et **une** maison :

```
      ses notes            les poids de           produit
                           cette maison
 ┌─────────────────┬──────────────────────┬──────────────┐
 │ Astronomy       │                      │              │
 │      -487,886   │   ×      0,001708    │    -0,8332   │
 │ Herbology       │                      │              │
 │         5,727   │   ×     -0,258183    │    -1,4787   │
 │ Défense         │                      │              │
 │         4,879   │   ×     -0,168558    │    -0,8224   │
 │ Ancient Runes   │                      │              │
 │       532,484   │   ×      0,021713    │   +11,5619   │
 │ Charms          │                      │              │
 │      -232,794   │   ×     -0,099092    │   +23,0680   │
 │ (biais)         │                      │              │
 │         1,000   │   ×    -37,501358    │   -37,5014   │
 └─────────────────┴──────────────────────┴──────────────┘
                                    somme  =   z = -6,0058
```

* La somme porte sur les **6 features d'un même élève**, jamais entre élèves.
* Le `1` du biais permet de traiter θ₀ comme un poids ordinaire.
  θ₀ est le **score de départ**, avant de regarder la moindre note : ici
  -37,5, soit « par défaut, ce n'est pas un Gryffindor ».
* Un poids **positif** rapproche de la maison, **négatif** en éloigne.
* ⚠️ Un **petit poids ≠ matière négligeable** : les poids absorbent les
  échelles (Astronomy a un petit poids car ses notes sont énormes).

## Étape B : appliquer la sigmoïde g()

```
            1
 g(z) = ──────────        z = -6,0058  →  g(z) = 0,0025
         1 + e⁻ᶻ
```

Elle replie `]-∞ ; +∞[` dans `]0 ; 1[` : le score devient une probabilité.

* `z = 0` → `0,5` : le point de bascule
* le **signe** de z décide, son **amplitude** dose la confiance
* elle n'atteint **jamais** exactement 0 ni 1
* ⚠️ le **signe moins** de l'exposant est essentiel : sans lui, la
  fonction est inversée et le modèle prédit systématiquement à l'envers

## Étape C : recommencer pour les 4 maisons, garder la plus haute

```
  x = [ -487,886 · 5,727 · 4,879 · 532,484 · -232,794 · 1 ]
   (les mêmes notes pour les 4 calculs)
        │
        ├── θ Gryffindor ──▶ z = -6,01 ──▶ g(z) = 0,0025
        ├── θ Hufflepuff ──▶ z = -3,20 ──▶ g(z) = 0,039
        ├── θ Ravenclaw  ──▶ z = +4,70 ──▶ g(z) = 0,991   ◀── le maximum
        └── θ Slytherin  ──▶ z = -5,40 ──▶ g(z) = 0,0045
                                              │
                                     argmax   ▼
                                        "Ravenclaw"
```

⚠️ **Ces 4 probabilités ne somment pas à 1** (ici 1,04). Les 4 classifieurs
ont été entraînés séparément et ne se concertent pas. Ce sont 4 scores de
confiance indépendants — d'où l'argmax, et non une lecture en pourcentages.

---

## Notations

| Symbole | Nom | C'est quoi |
|---|---|---|
| `x` | features | les 5 notes d'un élève + le `1` du biais |
| `θ` | poids (thêta) | les 6 nombres appris, **une ligne de theta.csv** |
| `θ₀` | biais | le poids qui ne multiplie aucune note |
| `θᵀx` | — | la somme pondérée : chaque note × son poids |
| `z` | — | juste un autre nom pour `θᵀx` |
| `g(z)` | sigmoïde / logistique | transforme `z` en probabilité |
| `hθ(x)` | hypothèse | `g(θᵀx)`, la prédiction du modèle |
| `y` | cible | 1 si l'élève est de cette maison, 0 sinon |
| `J(θ)` | coût / log-loss | l'erreur globale, à minimiser |
| `∂J/∂θⱼ` | dérivée partielle | la pente selon le poids `j` |
| `α` | learning rate | la taille du pas de correction |
| `m` | — | le nombre d'élèves (1600) |

---

## Questions probables en soutenance

* Pourquoi « régression » alors que c'est un classifieur ?
* Pourquoi la sigmoïde et pas un simple seuil ? → non dérivable, donc
  pas de pente à suivre, donc pas d'apprentissage possible.
* Pourquoi pas l'erreur quadratique ? → non convexe avec une sigmoïde.
* Pourquoi 4 classifieurs binaires plutôt qu'un modèle à 4 sorties ?
* Pourquoi les 4 probabilités ne font-elles pas 100 % ?
* Pourquoi standardiser les données avant d'entraîner ? → sinon la
  descente converge mal, et la sigmoïde sature immédiatement.
* Pourquoi le biais est-il ajouté **après** la standardisation ? → son
  écart-type vaut 0, on diviserait par zéro.
