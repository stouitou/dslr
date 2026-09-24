# 🧙 Logistic Regression — Hogwarts Houses

A machine learning project implementing **multiclass logistic regression from scratch**, using Python, NumPy and Pandas.

The goal is to predict a student's **Hogwarts House** based on their grades in different subjects.

---

## 📌 Project Goal

The objective is to build a complete machine learning pipeline without using a machine learning library for the logistic regression itself:

* explore and analyze the dataset
* visualize relationships between features
* train logistic regression models with relevant features
* predict Hogwarts Houses
* model evaluation

A **bonus implementation using Stochastic Gradient Descent (SGD)** is also included.

---

## ⚙️ How it works

The project is divided into several stages.

### 🔎 Data exploration

The first step is to understand the dataset and its features.

Several visualization scripts are provided:

* `describe` — displays statistical information about the numerical features
* `histogram` — displays feature distributions
* `pair_plot` — displays a scatter plot matrix to compare features
* `scatter_plot` — displays the relationship between two selected features

The pair plot is also used to identify features that appear to separate the four Hogwarts Houses.

---

### 🧹 Data preprocessing

Before training the models, the numerical data is prepared:

* missing values are replaced using the mean of each feature
* features are standardized
* a bias term is added to the input matrix

Standardization allows the different features to work on comparable scales and makes gradient descent more stable.

---

### 🧠 Training phase

The training script:

* loads the training dataset
* extracts the numerical features
* prepares the data
* trains one logistic regression model for each Hogwarts House
* optimizes the parameters using gradient descent
* saves the learned parameters to `theta.csv`

The project uses a **One-vs-Rest** strategy to handle the four Hogwarts Houses.

For each house, the model learns to distinguish:

> this house vs. all other houses

---

### 🔮 Prediction phase

The prediction script:

* loads the dataset
* loads the trained parameters from `theta.csv`
* computes the probability for each Hogwarts House
* selects the house with the highest probability
* saves the predictions to `houses.csv`

The prediction process can therefore be summarized as:

```text
Student
   ↓
Features
   ↓
4 logistic regression models
   ↓
4 probabilities
   ↓
Highest probability
   ↓
Predicted Hogwarts House
```

---

### 📊 Evaluation phase

Several tools are provided to evaluate and understand the model.

#### Accuracy

The `accuracy` script compares the predicted houses with the known houses and computes the classification accuracy.

#### Confusion matrix

The `confusion_matrix` script displays how the predictions are distributed between the four houses.

This makes it possible to see which houses are most often confused with each other.

#### Individual probabilities

The `personal_probabilities` script displays the probabilities produced by the four models for students whose predictions are less certain.

This helps visualize cases where the model has difficulty clearly separating one house from another.

---

## 🏰 One-vs-Rest Classification

Because logistic regression is a binary classifier, four separate models are trained.

For example:

```text
Model 1 → Gryffindor vs. all other houses
Model 2 → Hufflepuff vs. all other houses
Model 3 → Ravenclaw vs. all other houses
Model 4 → Slytherin vs. all other houses
```

During prediction, the four model outputs are compared and the house with the highest output is selected.

---

## 🎯 Feature Selection

Three visualizations were used to explore the dataset and identify relevant features for the logistic regression model:

* **Histogram:** identifies subjects with similar distributions across the four houses.
* **Scatter plot:** identifies subjects with similar behavior and relationships between their grades.
* **Pair plot:** shows how combinations of subjects can separate the students into four distinct groups corresponding to the Hogwarts Houses.

These visualizations helped identify potentially relevant features, including:

* Astronomy
* Herbology
* Ancient Runes

---

## 📈 Results

The project also includes tools to analyze the errors through:

* accuracy
* confusion matrix
* individual model probabilities

---

## 🚀 Bonus — Stochastic Gradient Descent

A bonus version of the logistic regression uses **Stochastic Gradient Descent (SGD)** instead of standard batch gradient descent.

The SGD implementation is located in:

```text
src/bonus_SGD/
```

It provides separate training and prediction scripts and generates its own parameter and prediction files:

```text
theta_sgd.csv
houses_SGD.csv
```

The purpose of this bonus is to explore another optimization method and compare it with the standard gradient descent implementation.

---

## 🛠️ Usage

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Train the model

```bash
./scripts/logreg_train data/dataset_train.csv
```

This generates:

```text
data/theta.csv
```

### Predict Hogwarts Houses

```bash
./scripts/logreg_predict data/dataset_test.csv data/theta.csv
```

This generates:

```text
data/houses.csv
```

### Evaluate the model

```bash
./scripts/accuracy data/dataset_train.csv data/houses.csv
```

A confusion matrix can also be generated with:

```bash
./scripts/confusion_matrix data/dataset_train.csv data/houses.csv
```

Other scripts are available for dataset exploration and visualization:

```bash
./scripts/describe
./scripts/histogram
./scripts/pair_plot
./scripts/scatter_plot
./scripts/personal_probabilities
```

See [`commands.md`](commands.md) for the complete list of commands and their arguments.

---

## 🧰 Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib

The logistic regression algorithm itself is implemented **from scratch**, without using a machine learning library.
