"""results/ 시각화 재현 스크립트 — scikit-learn + matplotlib
실행: python make_figures.py  (results/*.png 생성)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

R = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(R, exist_ok=True)
GREEN = "#76B900"
iris = load_iris()
X, y = iris.data[:, 2:4], iris.target

# SVM 결정경계
clf = SVC(kernel="rbf", C=1, gamma="scale").fit(X, y)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 400),
                     np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 400))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, alpha=.25, cmap="viridis")
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=35)
ax.set(xlabel="Petal length (cm)", ylabel="Petal width (cm)",
       title="SVM (RBF kernel) — Iris Decision Boundary")
fig.tight_layout(); fig.savefig(f"{R}/svm_decision_boundary.png", dpi=120); plt.close(fig)

# KNN 혼동행렬
Xtr, Xte, ytr, yte = train_test_split(iris.data, y, test_size=.3, random_state=42, stratify=y)
knn = KNeighborsClassifier(5).fit(Xtr, ytr)
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(knn, Xte, yte, display_labels=iris.target_names,
                                      cmap="Greens", ax=ax, colorbar=False)
ax.set_title(f"KNN (k=5) — Iris Confusion Matrix\naccuracy={knn.score(Xte, yte):.2f}")
fig.tight_layout(); fig.savefig(f"{R}/knn_confusion_matrix.png", dpi=120); plt.close(fig)

# KMeans 엘보우 + 클러스터 중심 (fruits_300.npy)
fr = np.load(os.path.join(os.path.dirname(__file__), "src", "KMeans_군집", "fruits_300.npy")).reshape(300, -1)
inertias = [KMeans(k, n_init=10, random_state=42).fit(fr).inertia_ for k in range(1, 8)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 8), inertias, "o-", color=GREEN); ax.axvline(3, ls="--", c="gray", alpha=.6)
ax.set(xlabel="k (clusters)", ylabel="Inertia", title="KMeans Elbow Method — fruits_300")
fig.tight_layout(); fig.savefig(f"{R}/kmeans_elbow.png", dpi=120); plt.close(fig)
km = KMeans(3, n_init=10, random_state=42).fit(fr)
fig, axs = plt.subplots(1, 3, figsize=(9, 3.3))
for i, ax in enumerate(axs):
    ax.imshow(km.cluster_centers_[i].reshape(100, 100), cmap="gray"); ax.axis("off"); ax.set_title(f"Cluster {i}")
fig.suptitle("KMeans Cluster Centers (mean fruit image)")
fig.tight_layout(); fig.savefig(f"{R}/kmeans_cluster_centers.png", dpi=120); plt.close(fig)

# 선형 회귀
db = load_diabetes(); xb = db.data[:, 2].reshape(-1, 1); yb = db.target
lr = LinearRegression().fit(xb, yb)
xs = np.linspace(xb.min(), xb.max(), 100).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(6, 4.3))
ax.scatter(xb, yb, s=14, alpha=.5, color="#4C72B0")
ax.plot(xs, lr.predict(xs), color=GREEN, lw=2.5, label=f"fit: R²={lr.score(xb, yb):.2f}")
ax.set(xlabel="BMI (normalized)", ylabel="Disease progression", title="Linear Regression — least squares fit")
ax.legend(); fig.tight_layout(); fig.savefig(f"{R}/linear_regression.png", dpi=120); plt.close(fig)
print("saved figures to", R)
