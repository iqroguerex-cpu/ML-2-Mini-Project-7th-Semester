import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Ensemble Playground", layout="wide")

st.title("Interactive Ensemble Learning Arena")
st.caption("Module 3: Decision by Committee — Boosting (AdaBoost, Stumping) vs. Bagging (Random Forests)")

# ----------------- SIDEBAR: DATASET & HYPERPARAMETERS -----------------
st.sidebar.header("1. Playground Dataset")
dataset_name = st.sidebar.selectbox("Select Pattern", ["Two Moons", "Concentric Circles", "Gaussian Blobs"])
n_samples = st.sidebar.slider("Number of Points", 100, 600, 300, step=50)
noise_level = st.sidebar.slider("Dataset Noise / Overlap", 0.05, 0.45, 0.20, step=0.05)

st.sidebar.header("2. Ensemble Parameters")
n_estimators = st.sidebar.slider("Number of Estimators (Trees)", 1, 100, 25, step=2)
adaboost_lr = st.sidebar.slider("AdaBoost Learning Rate", 0.1, 2.0, 1.0, step=0.1)
rf_depth = st.sidebar.slider("Random Forest Max Depth", 1, 15, 5, step=1)

# ----------------- DATA GENERATION -----------------
@st.cache_data
def get_data(name, n, noise):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=42)
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
    else:
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=noise * 8, random_state=42)
    return X, y

X, y = get_data(dataset_name, n_samples, noise_level)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- TRAIN ENSEMBLE MODELS -----------------
# AdaBoost uses Decision Stumps (max_depth=1) by default
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=adaboost_lr,
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)
ada_f1 = f1_score(y_test, ada_pred)

# Random Forest (Bagging with randomized feature subspaces)
rf_clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_depth,
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# ----------------- METRICS COMPARISON CARDS -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("AdaBoost Accuracy", f"{ada_acc * 100:.1f}%")
col1.caption(f"F1-Score: {ada_f1:.3f}")

col2.metric("Random Forest Accuracy", f"{rf_acc * 100:.1f}%")
col2.caption(f"F1-Score: {rf_f1:.3f}")

if ada_acc > rf_acc:
    col3.success(f"AdaBoost leading by +{(ada_acc - rf_acc) * 100:.1f}%")
elif rf_acc > ada_acc:
    col3.success(f"Random Forest leading by +{(rf_acc - ada_acc) * 100:.1f}%")
else:
    col3.info("Both ensembles tied in accuracy")

col4.metric("Dataset Size", f"{len(X_train)} Train / {len(X_test)} Test")

st.divider()

# ----------------- DECISION BOUNDARY PLOTS -----------------
st.subheader("Decision Boundary Visualization")
st.markdown(
    "Notice how **AdaBoost (Sequential Boosting)** builds sharp step-like boundaries using combinations of 1D decision stumps, "
    "while **Random Forest (Parallel Bagging)** produces smoother island-like contours."
)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_ada = ada_clf.predict(grid_points).reshape(xx.shape)
Z_rf = rf_clf.predict(grid_points).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot AdaBoost
ax1.contourf(xx, yy, Z_ada, alpha=0.35, cmap="coolwarm")
ax1.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", edgecolors="k", s=35, label="Class 0")
ax1.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", edgecolors="k", s=35, label="Class 1")
ax1.set_title(f"AdaBoost (Stumps = {n_estimators}) — Acc: {ada_acc * 100:.1f}%", fontsize=12, fontweight="bold")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend(loc="upper right")

# Plot Random Forest
ax2.contourf(xx, yy, Z_rf, alpha=0.35, cmap="coolwarm")
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", edgecolors="k", s=35, label="Class 0")
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", edgecolors="k", s=35, label="Class 1")
ax2.set_title(f"Random Forest (Trees = {n_estimators}) — Acc: {rf_acc * 100:.1f}%", fontsize=12, fontweight="bold")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend(loc="upper right")

plt.tight_layout()
st.pyplot(fig)

# ----------------- INTERACTIVE SINGLE-POINT PREDICTOR -----------------
st.subheader("Live Point Diagnostic")
st.markdown("Pick a custom coordinate $(X_1, X_2)$ to see how both models classify the point in real time:")

col_x, col_y, col_eval = st.columns([2, 2, 2])
pt_x = col_x.number_input("Feature 1 (X coordinate)", value=float(np.mean(X[:, 0])), format="%.2f")
pt_y = col_y.number_input("Feature 2 (Y coordinate)", value=float(np.mean(X[:, 1])), format="%.2f")

with col_eval:
    st.write(" ")
    st.write(" ")
    predict_btn = st.button("Classify Coordinate", type="primary")

if predict_btn:
    sample = np.array([[pt_x, pt_y]])
    ada_single_pred = ada_clf.predict(sample)[0]
    ada_single_prob = ada_clf.predict_proba(sample)[0]

    rf_single_pred = rf_clf.predict(sample)[0]
    rf_single_prob = rf_clf.predict_proba(sample)[0]

    res_col1, res_col2 = st.columns(2)
    res_col1.info(f"**AdaBoost Prediction:** Class `{ada_single_pred}` (Confidence: {ada_single_prob[ada_single_pred]*100:.1f}%)")
    res_col2.success(f"**Random Forest Prediction:** Class `{rf_single_pred}` (Confidence: {rf_single_prob[rf_single_pred]*100:.1f}%)")
