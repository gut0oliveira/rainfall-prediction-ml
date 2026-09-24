# Rainfall Prediction with Machine Learning

Can weather observations be used to predict whether it will rain the following day?

This project compares four supervised classification algorithms using historical weather observations from Sydney, Australia. It covers data acquisition, preprocessing, categorical encoding, model training, and evaluation with metrics suited to binary classification.

## Objective

Predict `RainTomorrow` from measurements such as temperature, humidity, pressure, wind, cloud cover, and current-day rainfall.

## Dataset

The dataset contains daily weather observations supplied for the IBM Machine Learning with Python final assignment. Its original meteorological data comes from the Australian Government Bureau of Meteorology.

The execution script downloads the dataset automatically and stores it locally in `data/raw/`, which is excluded from version control.

## Methodology

The reproducible pipeline:

1. separates features from the target;
2. uses a stratified 80/20 train-test split;
3. imputes missing numerical and categorical values;
4. standardizes numerical variables;
5. one-hot encodes categorical variables;
6. trains and compares four classifiers.

Models evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

The comparison uses accuracy, precision, recall, F1-score, and Jaccard index. F1-score is the primary ranking metric because rainfall prediction involves an imbalanced target and both false positives and false negatives matter.

## Results

The evaluation used a fixed random seed and the same stratified test set for every model:

| Model | Accuracy | Precision | Recall | F1-score | Jaccard |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8168 | 0.6689 | 0.5824 | **0.6226** | **0.4521** |
| Support Vector Machine | 0.8168 | **0.6894** | 0.5353 | 0.6026 | 0.4313 |
| K-Nearest Neighbors | 0.8046 | 0.6944 | 0.4412 | 0.5396 | 0.3695 |
| Decision Tree | 0.7786 | 0.5984 | 0.4471 | 0.5118 | 0.3439 |

Logistic Regression produced the best balance between precision and recall, reaching the highest F1-score and Jaccard index. KNN achieved slightly higher precision, but its lower recall means it missed more rainy days. This comparison also shows why accuracy alone is not sufficient to select a model for an imbalanced classification problem.

Running the project reproduces this evaluation and generates the full table in `reports/model_metrics.csv`.

## Project structure

```text
rainfall-prediction-ml/
├── data/
│   └── raw/                 # downloaded dataset (not tracked)
├── notebooks/
│   └── rainfall_prediction.ipynb
├── reports/                 # generated model comparison
├── src/
│   └── main.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and run the complete pipeline:

```bash
pip install -r requirements.txt
python src/main.py
```

## Study notebook

The notebook in `notebooks/` is retained as the original learning artifact based on IBM Skills Network course material. The production-style script is the corrected, reproducible version used for the project results.

## Technologies

Python · Pandas · scikit-learn · Jupyter Notebook

## License

This repository is available under the [MIT License](LICENSE). Third-party course material and the dataset remain subject to their respective attribution and terms.
