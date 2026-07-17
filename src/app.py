from utils import db_connect
engine = db_connect()

# your code here
"""
Proyecto: Predicción de Diabetes con Árbol de Decisión
Dataset: Pima Indians Diabetes (diabetes__2_.csv)
Autor: Gaspar Diaz - 4Geeks Academy

Pipeline: EDA -> deteccion de ceros invalidos -> split -> imputacion (sin fuga de datos)
          -> arbol baseline -> poda con GridSearchCV -> evaluacion -> conclusiones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
df = pd.read_csv("diabetes__2_.csv")

print("Forma del dataset:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nResumen estadistico:\n", df.describe())


# ---------------------------------------------------------------------------
# 2. Deteccion de ceros invalidos (faltantes ocultos)
# ---------------------------------------------------------------------------
cols_sospechosas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zeros_resumen = pd.DataFrame({
    "n_ceros": (df[cols_sospechosas] == 0).sum(),
    "pct_ceros": (df[cols_sospechosas] == 0).mean() * 100
})
print("\nCeros invalidos por columna:\n", zeros_resumen)


# ---------------------------------------------------------------------------
# 3. Indicadores de faltante + reemplazo de ceros por NaN
#    (esto NO usa la variable Outcome, por lo que es seguro hacerlo
#    antes del split)
# ---------------------------------------------------------------------------
df["SkinThickness_missing"] = (df["SkinThickness"] == 0).astype(int)
df["Insulin_missing"] = (df["Insulin"] == 0).astype(int)

df[cols_sospechosas] = df[cols_sospechosas].replace(0, np.nan)


# ---------------------------------------------------------------------------
# 4. EDA visual (sobre el dataset completo, solo exploratorio)
# ---------------------------------------------------------------------------
# 4.1 Balance de clases
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="Outcome", data=df, ax=ax)
ax.set_title("Balance de clases (Outcome)")
ax.set_xlabel("Outcome (0 = No diabetes, 1 = Diabetes)")
plt.tight_layout()
plt.savefig("eda_balance_clases.png", dpi=120)
plt.close(fig)

print("\nProporcion de clases:\n", df["Outcome"].value_counts(normalize=True) * 100)

# 4.2 Distribucion de variables numericas por Outcome
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.kdeplot(data=df, x=col, hue="Outcome", fill=True, ax=axes[i], common_norm=False)
    axes[i].set_title(f"Distribucion de {col} por Outcome")
plt.tight_layout()
plt.savefig("eda_distribuciones.png", dpi=120)
plt.close(fig)

# 4.3 Matriz de correlacion
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.drop(columns=["Outcome"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlacion")
plt.tight_layout()
plt.savefig("eda_correlacion.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Split train/test (ANTES de imputar, para evitar fuga de datos)
# ---------------------------------------------------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain:", X_train.shape, "Test:", X_test.shape)


# ---------------------------------------------------------------------------
# 6. Imputacion con medianas calculadas SOLO en train
#    (sin condicionar a Outcome, para no filtrar informacion de la etiqueta)
# ---------------------------------------------------------------------------
medianas_train = X_train[cols_sospechosas].median()
print("\nMedianas calculadas en train:\n", medianas_train)

X_train[cols_sospechosas] = X_train[cols_sospechosas].fillna(medianas_train)
X_test[cols_sospechosas] = X_test[cols_sospechosas].fillna(medianas_train)

assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 7. Modelo baseline: Arbol de Decision con hiperparametros por defecto
# ---------------------------------------------------------------------------
tree_baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_baseline.fit(X_train, y_train)

print("\n--- BASELINE ---")
print("Profundidad:", tree_baseline.get_depth(), "| Hojas:", tree_baseline.get_n_leaves())

y_train_pred = tree_baseline.predict(X_train)
y_test_pred = tree_baseline.predict(X_test)
y_test_proba = tree_baseline.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (baseline) ===")
print(classification_report(y_train, y_train_pred))
print("\n=== TEST (baseline) ===")
print(classification_report(y_test, y_test_pred))
print("AUC test (baseline):", roc_auc_score(y_test, y_test_proba))

importancias_baseline = pd.Series(
    tree_baseline.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (baseline):\n", importancias_baseline)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de confusion - Baseline (test)")
plt.tight_layout()
plt.savefig("confusion_baseline.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Poda de hiperparametros (grid pequeno + validacion cruzada)
# ---------------------------------------------------------------------------
param_grid = {
    "max_depth": [3, 4, 5, 6, 7],
    "min_samples_leaf": [5, 10, 20],
    "class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n--- GRID SEARCH ---")
print("Mejores hiperparametros:", grid_search.best_params_)
print("Mejor F1 (CV):", grid_search.best_score_)

tree_tuned = grid_search.best_estimator_
print("Profundidad podado:", tree_tuned.get_depth(), "| Hojas:", tree_tuned.get_n_leaves())


# ---------------------------------------------------------------------------
# 9. Evaluacion del modelo final (podado)
# ---------------------------------------------------------------------------
y_train_pred_tuned = tree_tuned.predict(X_train)
y_test_pred_tuned = tree_tuned.predict(X_test)
y_test_proba_tuned = tree_tuned.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (podado) ===")
print(classification_report(y_train, y_train_pred_tuned))
print("\n=== TEST (podado) ===")
print(classification_report(y_test, y_test_pred_tuned))
print("AUC test (podado):", roc_auc_score(y_test, y_test_proba_tuned))

importancias_tuned = pd.Series(
    tree_tuned.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (podado):\n", importancias_tuned)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_tuned),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de confusion - Podado (test)")
plt.tight_layout()
plt.savefig("confusion_podado.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    tree_tuned,
    feature_names=X_train.columns,
    class_names=["No diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
plt.tight_layout()
plt.savefig("arbol_podado.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Random Forest baseline (hiperparametros por defecto)
# ---------------------------------------------------------------------------
rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_baseline.fit(X_train, y_train)

print("\n--- RANDOM FOREST BASELINE ---")
print("Numero de arboles:", rf_baseline.n_estimators)
print("Profundidad promedio:",
      np.mean([tree.get_depth() for tree in rf_baseline.estimators_]))

y_train_pred_rf = rf_baseline.predict(X_train)
y_test_pred_rf = rf_baseline.predict(X_test)
y_test_proba_rf = rf_baseline.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (RF baseline) ===")
print(classification_report(y_train, y_train_pred_rf))
print("\n=== TEST (RF baseline) ===")
print(classification_report(y_test, y_test_pred_rf))
print("AUC test (RF baseline):", roc_auc_score(y_test, y_test_proba_rf))

importancias_rf = pd.Series(
    rf_baseline.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (RF baseline):\n", importancias_rf)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_rf),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Greens")
ax.set_title("Matriz de confusion - RF baseline (test)")
plt.tight_layout()
plt.savefig("confusion_rf_baseline.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Ajuste de Random Forest (grid pequeno + validacion cruzada)
# ---------------------------------------------------------------------------
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [5, 10],
    "class_weight": [None, "balanced"]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid_rf,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train)

print("\n--- GRID SEARCH (Random Forest) ---")
print("Mejores hiperparametros RF:", grid_search_rf.best_params_)
print("Mejor F1 (CV):", grid_search_rf.best_score_)

rf_tuned = grid_search_rf.best_estimator_


# ---------------------------------------------------------------------------
# 12. Evaluacion del Random Forest ajustado (modelo final)
# ---------------------------------------------------------------------------
y_train_pred_rf_tuned = rf_tuned.predict(X_train)
y_test_pred_rf_tuned = rf_tuned.predict(X_test)
y_test_proba_rf_tuned = rf_tuned.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (RF ajustado) ===")
print(classification_report(y_train, y_train_pred_rf_tuned))
print("\n=== TEST (RF ajustado) ===")
print(classification_report(y_test, y_test_pred_rf_tuned))
print("AUC test (RF ajustado):", roc_auc_score(y_test, y_test_proba_rf_tuned))

importancias_rf_tuned = pd.Series(
    rf_tuned.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (RF ajustado):\n", importancias_rf_tuned)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_rf_tuned),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Greens")
ax.set_title("Matriz de confusion - RF ajustado (test)")
plt.tight_layout()
plt.savefig("confusion_rf_ajustado.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
importancias_rf_tuned.plot(kind="barh", ax=ax, color="seagreen")
ax.set_title("Importancia de variables - RF ajustado")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("importancia_rf_ajustado.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 13. XGBoost baseline (hiperparametros por defecto)
# ---------------------------------------------------------------------------
xgb_baseline = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
xgb_baseline.fit(X_train, y_train)

y_train_pred_xgb = xgb_baseline.predict(X_train)
y_test_pred_xgb = xgb_baseline.predict(X_test)
y_test_proba_xgb = xgb_baseline.predict_proba(X_test)[:, 1]

print("\n--- XGBOOST BASELINE ---")
print("\n=== TRAIN (XGBoost baseline) ===")
print(classification_report(y_train, y_train_pred_xgb))
print("\n=== TEST (XGBoost baseline) ===")
print(classification_report(y_test, y_test_pred_xgb))
print("AUC test (XGBoost baseline):", roc_auc_score(y_test, y_test_proba_xgb))

importancias_xgb = pd.Series(
    xgb_baseline.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (XGBoost baseline):\n", importancias_xgb)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_xgb),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Oranges")
ax.set_title("Matriz de confusion - XGBoost baseline (test)")
plt.tight_layout()
plt.savefig("confusion_xgb_baseline.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 14. Ajuste de XGBoost (grid pequeno + validacion cruzada)
# ---------------------------------------------------------------------------
# scale_pos_weight recomendado = ratio de clase negativa / clase positiva (en train)
ratio_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("\nscale_pos_weight sugerido:", ratio_pos_weight)

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.05, 0.1],
    "scale_pos_weight": [1, ratio_pos_weight]
}

grid_search_xgb = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    param_grid_xgb,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search_xgb.fit(X_train, y_train)

print("\n--- GRID SEARCH (XGBoost) ---")
print("Mejores hiperparametros XGBoost:", grid_search_xgb.best_params_)
print("Mejor F1 (CV):", grid_search_xgb.best_score_)

xgb_tuned = grid_search_xgb.best_estimator_


# ---------------------------------------------------------------------------
# 15. Evaluacion del XGBoost ajustado
# ---------------------------------------------------------------------------
y_train_pred_xgb_tuned = xgb_tuned.predict(X_train)
y_test_pred_xgb_tuned = xgb_tuned.predict(X_test)
y_test_proba_xgb_tuned = xgb_tuned.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (XGBoost ajustado) ===")
print(classification_report(y_train, y_train_pred_xgb_tuned))
print("\n=== TEST (XGBoost ajustado) ===")
print(classification_report(y_test, y_test_pred_xgb_tuned))
print("AUC test (XGBoost ajustado):", roc_auc_score(y_test, y_test_proba_xgb_tuned))

importancias_xgb_tuned = pd.Series(
    xgb_tuned.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (XGBoost ajustado):\n", importancias_xgb_tuned)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_xgb_tuned),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Oranges")
ax.set_title("Matriz de confusion - XGBoost ajustado (test)")
plt.tight_layout()
plt.savefig("confusion_xgb_ajustado.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
importancias_xgb_tuned.plot(kind="barh", ax=ax, color="darkorange")
ax.set_title("Importancia de variables - XGBoost ajustado")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("importancia_xgb_ajustado.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 16. Resumen final en consola: comparacion de los cuatro modelos
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN FINAL - COMPARACION DE MODELOS")
print("=" * 60)

def metricas_clase_1(y_true, y_pred):
    reporte = classification_report(y_true, y_pred, output_dict=True)
    return reporte["1"]["precision"], reporte["1"]["recall"], reporte["1"]["f1-score"], reporte["accuracy"]

prec_arbol, rec_arbol, f1_arbol, acc_arbol = metricas_clase_1(y_test, y_test_pred_tuned)
prec_rf, rec_rf, f1_rf, acc_rf = metricas_clase_1(y_test, y_test_pred_rf_tuned)
prec_xgb, rec_xgb, f1_xgb, acc_xgb = metricas_clase_1(y_test, y_test_pred_xgb_tuned)

resumen = pd.DataFrame({
    "Arbol podado": [acc_arbol, f1_arbol, rec_arbol, prec_arbol,
                      roc_auc_score(y_test, y_test_proba_tuned)],
    "RF baseline": [
        classification_report(y_test, y_test_pred_rf, output_dict=True)["accuracy"],
        classification_report(y_test, y_test_pred_rf, output_dict=True)["1"]["f1-score"],
        classification_report(y_test, y_test_pred_rf, output_dict=True)["1"]["recall"],
        classification_report(y_test, y_test_pred_rf, output_dict=True)["1"]["precision"],
        roc_auc_score(y_test, y_test_proba_rf),
    ],
    "RF ajustado": [acc_rf, f1_rf, rec_rf, prec_rf,
                    roc_auc_score(y_test, y_test_proba_rf_tuned)],
    "XGBoost ajustado": [acc_xgb, f1_xgb, rec_xgb, prec_xgb,
                         roc_auc_score(y_test, y_test_proba_xgb_tuned)],
}, index=["Accuracy", "F1 (Diabetes)", "Recall (Diabetes)", "Precision (Diabetes)", "AUC"])

print(resumen.round(3))

print(f"\nModelo con mejor accuracy: XGBoost ajustado {grid_search_xgb.best_params_}")
print("Variables mas importantes (XGBoost ajustado):")
print(importancias_xgb_tuned.head(3))
print("""
Nota metodologica: la imputacion de valores faltantes se realizo usando
UNICAMENTE estadisticos (medianas) calculados sobre el conjunto de train,
sin condicionar a la variable Outcome, para evitar fuga de datos hacia
el modelo. XGBoost ajustado (max_depth=3, learning_rate=0.05,
scale_pos_weight ~1.87) logra el mejor accuracy y F1 de los cuatro
modelos evaluados, aunque con una mejora marginal frente al Random
Forest ajustado.
""")