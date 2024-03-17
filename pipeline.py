import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models import get_models
from fit import fit_model
from train_save_model import save_model_to_onnx
from visualize import save_confusion_matrices, save_roc_curves
import numpy as np
from tqdm import tqdm
df = pd.read_json('data.json')

X = df['text']
y = df['label']

models = get_models()
model_scores = {}
confusion_matrices = {}
roc_curves = {}
best_score = 0
best_model = None
best_model_name = "None"

for model_name, model in tqdm(models.items(), desc="Models"):
    tqdm.write(f'Fitting and evaluating {model_name}...')
    score = fit_model(model, X,y)
    tqdm.write(f'Cross-validated accuracy: {score}')
    model_scores[model_name] = score
    
    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = model_name

    confusion_matrices[model_name] = np.zeros((2,2))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in StratifiedKFold(n_splits=5).split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        confusion_matrices[model_name] += confusion_matrix(y_test, y_pred)
        y_score = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        roc_curves[model_name] = (mean_fpr, mean_tpr, mean_auc)

save_confusion_matrices(confusion_matrices)
save_roc_curves(roc_curves)

save_model_to_onnx(best_model)
print(f"Best Model: {best_model_name} with score {best_score}")

