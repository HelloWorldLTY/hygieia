import pandas as pd 
import scanpy as sc

df1 = pd.read_csv("./disease_diagnosis_common.csv") # https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset
df2 = pd.read_excel("./Rare_diseases_842_final.xlsx") # https://www.nature.com/articles/s41598-025-90450-0
pheno_list = []
for i in df1.index:
    row = df1.loc[i]
    i1,i2,i3 = row['Symptom_1'], row['Symptom_2'], row['Symptom_3']
    
    pheno_list_item = ', '.join([i1.lower(),i2.lower(),i3.lower()])
    pheno_list_item += '.'
    
    pheno_list.append(pheno_list_item)
    

df1['phenotype'] = pheno_list
common_id = ['common' if i != 'Healthy' else 'healthy' for i in df1['Diagnosis'].values]
df1['diagnosis_type'] = common_id

df2['clean_pheno'] = ['.'.join(i.split('.')[1:]) for i in df2['diagnosis'].values]


df = pd.concat((df1,df2))
df.to_csv("phenotype_rare_common_diagnosis.csv")

df = pd.read_csv("phenotype_rare_common_diagnosis.csv",index_col=0)

# Loading data from saved file
import json
results = []
with open(f"./phenotype_diagnosis_gpt5_embeddings.jsonl", 'r') as file: #get phenotye embeddings
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)
#         break

alldata_output = []
for index in range(len(results)):
    out = results[index]['response']['body']['data'][0]['embedding']
#     print(out)
    alldata_output.append(out)

import numpy as np
np.array(alldata_output).shape

# alldata_output[0]

df

out_data = np.array(alldata_output)

sampled_data = df.sample(n=400, random_state=2024,replace=False)

train_data = df.loc[~df.index.isin(sampled_data.index)] 

len(train_data)

# sampled_data['label'].value_counts()

# train_ensemble.py
# scikit-learn ensemble learning: training, tuning, and evaluating classifiers

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)

# Base learners & ensembles
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)

RANDOM_STATE = 42

# -----------------------
# 1) Load data
# -----------------------
X_train = out_data[train_data.index]
y_train = train_data['label'].values

X_test = out_data[sampled_data.index]
y_test = sampled_data['label'].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# -----------------------
# 2) Define base models (use pipelines when scaling helps)
# -----------------------
logreg = Pipeline([
    ("clf", LogisticRegression(max_iter=1000, multi_class="auto", solver="lbfgs", random_state=RANDOM_STATE))
])

svc = Pipeline([
    ("clf", SVC(probability=True, random_state=RANDOM_STATE))
])

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)

# -----------------------
# 3) Quick tuning for a few strong performers
#    (Keep grids small to stay fast—expand as needed.)
# -----------------------
param_grid_rf = {
    "n_estimators": [100,200, 400],
    "max_depth": [None, 8, 16],
    "min_samples_split": [2, 5]
}

param_grid_hgb = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [None, 6, 10],
    "l2_regularization": [0.0, 0.1]
}

param_grid_svc = {
    "clf__C": [0.5, 1.0, 2.0],
    "clf__gamma": ["scale", "auto"]
}

def tune(model, param_grid, X, y, name: str):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    gs.fit(X, y)
    print(f"[{name}] best params: {gs.best_params_}")
    print(f"[{name}] best CV AUC: {gs.best_score_:.4f}\n")
    return gs.best_estimator_

rf_best = tune(rf, param_grid_rf, X_train, y_train, "RandomForest")
hgb_best = tune(hgb, param_grid_hgb, X_train, y_train, "HistGB")
svc_best = tune(svc, param_grid_svc, X_train, y_train, "SVC")

# We’ll keep a simple, well-behaved logistic regression without tuning for diversity
logreg_best = logreg.fit(X_train, y_train)

# -----------------------
# 4) Build ensembles
# -----------------------

# 4a) Soft Voting (averages predicted probabilities)
voting = VotingClassifier(
    estimators=[
        ("rf", rf_best),
        ("hgb", hgb_best),
        ("svc", svc_best),
        ("lr", logreg_best)
    ],
    voting="soft",  # better for calibrated probabilistic outputs
    n_jobs=-1
)

# 4b) Stacking (meta-learner combines base learners’ predictions)
# Note: meta learner gets out-of-fold predictions from base estimators during fit
stacking = StackingClassifier(
    estimators=[
        ("rf", rf_best),
        ("hgb", hgb_best),
        ("svc", svc_best),
        ("lr", logreg_best)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    passthrough=False,            # set True to include raw features as well
    cv=cv,
    n_jobs=-1
)

# 4c) Classic Bagging around a strong but high-variance base (SVC w/ scaling)
bagging_svc = BaggingClassifier(
    base_estimator=svc_best,
    n_estimators=100,
    max_samples=0.9,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# -----------------------
# 5) Cross-validate ensembles
# -----------------------
def cv_report(name: str, model) -> None:
    auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name}: CV AUC {auc.mean():.4f} ± {auc.std():.4f} | "
          f"CV Acc {acc.mean():.4f} ± {acc.std():.4f}")

for name, model in [
    ("Voting", voting),
    ("Stacking", stacking),
    ("Bagging(SVC)", bagging_svc),
    ("RandomForest(best)", rf_best),
    ("HistGB(best)", hgb_best)
]:
    cv_report(name, model)
print()

# -----------------------
# 6) Fit best ensemble(s) and evaluate on the test set
#    (Here we show both Voting and Stacking.)
# -----------------------
def evaluate(name: str, model, X_tr, y_tr, X_te, y_te) -> Dict[str, Any]:
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_te)

    metrics = {
        "ACC": accuracy_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
    }
    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_te, y_prob)

    print(f"=== {name} Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print("\nClassification Report:\n", classification_report(y_te, y_pred))
    return metrics

# _ = evaluate("Voting", voting, X_train, y_train, X_test, y_test)
# _ = evaluate("Stacking", stacking, X_train, y_train, X_test, y_test)

# # -----------------------
# # 7) (Optional) Single strong tuned model baseline
# # -----------------------
# _ = evaluate("HistGB(best)", hgb_best, X_train, y_train, X_test, y_test)


def evaluate(name: str, model, X_tr, y_tr, X_te, y_te) -> Dict[str, Any]:
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = model.predict(X_te)

    metrics = {
        "ACC": accuracy_score(y_te, y_pred)
    }
#     if y_prob is not None:
#         metrics["ROC_AUC"] = roc_auc_score(y_te, y_prob)

    print(f"=== {name} Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(y_te, y_pred))
    return metrics

_ = evaluate("Stacking", stacking, X_train, y_train, X_test, y_test)
_ = evaluate("HistGB(best)", hgb_best, X_train, y_train, X_test, y_test)
_ = evaluate("Voting", voting, X_train, y_train, X_test, y_test)

import pickle
filename = "Stacking_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(stacking, file)
    
import pickle
filename = "Hgb_best_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(hgb_best, file)

import pickle
filename = "Voting_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(voting, file)