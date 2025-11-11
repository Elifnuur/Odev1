# Breast Cancer (k-NN + Pipeline + GridSearchCV) — malignant(0) pozitif
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# 1) Veri
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # 0=malignant(+) , 1=benign

# 2) Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=65, stratify=y
)

# 3) Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# 4) Hiperparametre ızgarası (makul ve yeterli)
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],  # 1=Manhattan, 2=Euclidean
}

# 5) AUC skoru: malignant(0) pozitif kabulü
def auc_malignant_pos(estimator, Xv, yv):
    p0 = estimator.predict_proba(Xv)[:, 0]  # sınıf 0 (malignant) olasılığı
    return roc_auc_score(yv, p0)            # pos_label=0 tutarlı

# 6) GridSearchCV (cv=5, n_jobs=1 → macOS semlock uyarılarını önler)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=auc_malignant_pos,
    cv=5,
    n_jobs=1,
    refit=True
)

grid.fit(X_train, y_train)
best = grid.best_estimator_

print("Best params:", grid.best_params_)
print(f"Best CV ROC-AUC (malignant=0): {grid.best_score_:.3f}")

# 7) Test değerlendirme
y_pred = best.predict(X_test)
p0_test = best.predict_proba(X_test)[:, 0]        # malignant(0) olasılığı
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, p0_test)              # pos_label=0 ile uyumlu
print(f"Test Accuracy: {acc:.3f}")
print(f"Test ROC-AUC (malignant=0): {auc:.3f}")

# 8) Görselleştirme
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["malignant(0)", "benign(1)"]
)
plt.title("Confusion Matrix (Best k-NN)")
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_test, p0_test, pos_label=0)
plt.title("ROC Curve (Best k-NN, malignant=positive)")
plt.tight_layout()
plt.show()
