import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Interactive Ensemble Arena", layout="wide")

# ----------------- HEADER & ARCHITECTURE GUIDE -----------------
st.title("Interactive Ensemble Learning Arena")
st.markdown("### Module 3: Decision by Committee — Boosting vs. Bagging")

with st.expander("📖 Read Theoretical Guide (How it works under the hood)", expanded=False):
    theory_col1, theory_col2 = st.columns(2)
    with theory_col1:
        st.markdown("""
        #### 1. Bagging: Random Forest
        * **Paradigm:** Parallel execution.
        * **Base Learner:** Deep, fully-grown decision trees.
        * **Mechanism:**
          * Samples data points with replacement (**Bootstrapping**).
          * Evaluates random feature subsets at every split.
          * Aggregates predictions via **unweighted majority voting**.
        * **Primary Role:** Reduces **Variance** (prevents overfitting).
        """)
    with theory_col2:
        st.markdown("""
        #### 2. Boosting: AdaBoost
        * **Paradigm:** Sequential execution.
        * **Base Learner:** Decision Stumps (1-split trees: $\\text{depth} = 1$).
        * **Mechanism:**
          * Trains a stump, evaluates errors.
          * Increases the sample weights of **misclassified** instances.
          * Aggregates predictions via **accuracy-weighted voting** ($\\alpha_t$).
        * **Primary Role:** Reduces **Bias** (turns weak learners into strong).
        """)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("1. Dataset Config")
dataset_name = st.sidebar.selectbox("Pattern Shape", ["Two Moons", "Concentric Circles", "Gaussian Blobs"])
n_samples = st.sidebar.slider("Sample Count", 100, 600, 300, step=50)
noise_level = st.sidebar.slider("Noise / Overlap", 0.05, 0.40, 0.20, step=0.05)

st.sidebar.header("2. Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators (Trees/Stumps)", 1, 100, 25, step=2)
adaboost_lr = st.sidebar.slider("AdaBoost Learning Rate (η)", 0.1, 2.0, 1.0, step=0.1)
rf_depth = st.sidebar.slider("Random Forest Max Depth", 1, 12, 5, step=1)

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

# ----------------- MODEL TRAINING -----------------
# 1. AdaBoost with Decision Stumps
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

# 2. Random Forest (Bagging)
rf_clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_depth,
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# ----------------- PERFORMANCE METRICS -----------------
st.subheader("1. Comparative Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("AdaBoost Accuracy", f"{ada_acc * 100:.1f}%")
m1.caption(f"F1-Score: {ada_f1:.3f}")

m2.metric("Random Forest Accuracy", f"{rf_acc * 100:.1f}%")
m2.caption(f"F1-Score: {rf_f1:.3f}")

if ada_acc > rf_acc:
    m3.success(f"AdaBoost leads by +{(ada_acc - rf_acc) * 100:.1f}%")
elif rf_acc > ada_acc:
    m3.success(f"Random Forest leads by +{(rf_acc - ada_acc) * 100:.1f}%")
else:
    m3.info("Models are tied in accuracy")

m4.metric("Test Evaluation Set", f"{len(X_test)} data points")

# ----------------- DECISION BOUNDARY PLOTS -----------------
st.subheader("2. Decision Boundary Topography")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_ada = ada_clf.predict(grid_points).reshape(xx.shape)
Z_rf = rf_clf.predict(grid_points).reshape(xx.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# AdaBoost Subplot
ax1.contourf(xx, yy, Z_ada, alpha=0.35, cmap="coolwarm")
ax1.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", edgecolors="k", s=35, label="Class 0 (Blue)")
ax1.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", edgecolors="k", s=35, label="Class 1 (Red)")
ax1.set_title(f"AdaBoost (Stumps = {n_estimators})\nBoundary: Sharp, orthogonal axis cuts", fontsize=11, fontweight="bold")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.legend(loc="upper right")

# Random Forest Subplot
ax2.contourf(xx, yy, Z_rf, alpha=0.35, cmap="coolwarm")
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color="blue", edgecolors="k", s=35, label="Class 0 (Blue)")
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color="red", edgecolors="k", s=35, label="Class 1 (Red)")
ax2.set_title(f"Random Forest (Trees = {n_estimators}, Depth = {rf_depth})\nBoundary: Smooth, averaged contours", fontsize=11, fontweight="bold")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
ax2.legend(loc="upper right")

plt.tight_layout()
st.pyplot(fig)

# ----------------- REAL-TIME POINT CLASSIFICATION -----------------
st.subheader("3. Live Inference Inspector")
st.markdown("Pick a custom coordinate $(X_1, X_2)$ to test how both ensembles classify the point:")

c_x, c_y, c_btn = st.columns([2, 2, 2])
pt_x = c_x.number_input("Feature 1 (X)", value=float(np.mean(X[:, 0])), format="%.2f")
pt_y = c_y.number_input("Feature 2 (Y)", value=float(np.mean(X[:, 1])), format="%.2f")

with c_btn:
    st.write(" ")
    st.write(" ")
    evaluate = st.button("Classify Coordinate", type="primary")

if evaluate:
    sample = np.array([[pt_x, pt_y]])
    ada_single_pred = ada_clf.predict(sample)[0]
    ada_single_prob = ada_clf.predict_proba(sample)[0]

    rf_single_pred = rf_clf.predict(sample)[0]
    rf_single_prob = rf_clf.predict_proba(sample)[0]

    r1, r2 = st.columns(2)
    r1.info(f"**AdaBoost Outcome:** Class `{ada_single_pred}` (Confidence: {ada_single_prob[ada_single_pred]*100:.1f}%)")
    r2.success(f"**Random Forest Outcome:** Class `{rf_single_pred}` (Confidence: {rf_single_prob[rf_single_pred]*100:.1f}%)")
