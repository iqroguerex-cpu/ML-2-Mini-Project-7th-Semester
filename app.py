import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Ensemble Playground (Seaborn)", layout="wide")

# Apply modern Seaborn theme globally
sns.set_theme(style="whitegrid", palette="muted")

st.title("Interactive Ensemble Learning Arena")
st.caption("Module 3: Decision by Committee — Boosting (AdaBoost) vs. Bagging (Random Forest)")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("1. Dataset Configuration")
dataset_name = st.sidebar.selectbox("Pattern Shape", ["Two Moons", "Concentric Circles", "Gaussian Blobs"])
n_samples = st.sidebar.slider("Sample Count", 100, 800, 350, step=50)
noise_level = st.sidebar.slider("Noise / Overlap", 0.05, 0.40, 0.20, step=0.05)

st.sidebar.header("2. Hyperparameters")
n_estimators = st.sidebar.slider("Number of Estimators (Trees/Stumps)", 5, 120, 35, step=5)
adaboost_lr = st.sidebar.slider("AdaBoost Learning Rate (η)", 0.1, 2.0, 1.0, step=0.1)
rf_depth = st.sidebar.slider("Random Forest Max Depth", 1, 12, 5, step=1)

# ----------------- DATASET SYNTHESIS -----------------
@st.cache_data
def generate_data(name, n, noise):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=42)
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
    else:
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=noise * 8, random_state=42)
    return X, y

X, y = generate_data(dataset_name, n_samples, noise_level)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_test = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df_test["Target"] = y_test.astype(str)

# ----------------- MODEL TRAINING -----------------
# AdaBoost with Decision Stumps
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=adaboost_lr,
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_preds = ada_clf.predict(X_test)
ada_probs = ada_clf.predict_proba(X_test)[:, 1]
ada_acc = accuracy_score(y_test, ada_preds)
ada_f1 = f1_score(y_test, ada_preds)

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=rf_depth,
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
rf_probs = rf_clf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)

# ----------------- METRICS CARDS -----------------
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
    m3.info("Models tied in Accuracy")

m4.metric("Evaluation Samples", f"{len(X_test)} points")

st.divider()

# ----------------- VISUALIZATION 1: DECISION BOUNDARIES -----------------
st.subheader("1. Spatial Decision Boundary Topography")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_ada = ada_clf.predict(grid_points).reshape(xx.shape)
Z_rf = rf_clf.predict(grid_points).reshape(xx.shape)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# AdaBoost boundary
ax1.contourf(xx, yy, Z_ada, alpha=0.3, cmap="coolwarm")
sns.scatterplot(
    data=df_test, x="Feature 1", y="Feature 2", hue="Target",
    palette={"0": "royalblue", "1": "crimson"},
    ax=ax1, edgecolor="black", s=45, alpha=0.9
)
ax1.set_title(f"AdaBoost (Stumps = {n_estimators}) — Acc: {ada_acc * 100:.1f}%", fontweight="bold")

# Random Forest boundary
ax2.contourf(xx, yy, Z_rf, alpha=0.3, cmap="coolwarm")
sns.scatterplot(
    data=df_test, x="Feature 1", y="Feature 2", hue="Target",
    palette={"0": "royalblue", "1": "crimson"},
    ax=ax2, edgecolor="black", s=45, alpha=0.9
)
ax2.set_title(f"Random Forest (Trees = {n_estimators}, Depth = {rf_depth}) — Acc: {rf_acc * 100:.1f}%", fontweight="bold")

plt.tight_layout()
st.pyplot(fig1)

# ----------------- VISUALIZATION 2: ENSEMBLE LEARNING STAGES -----------------
st.subheader("2. Model Staged Accuracy vs. Estimator Progression")

ada_staged_acc = [accuracy_score(y_test, p) for p in ada_clf.staged_predict(X_test)]
# Evaluate progressive sub-forest accuracy
rf_staged_acc = []
for i in range(1, n_estimators + 1):
    sub_rf = RandomForestClassifier(n_estimators=i, max_depth=rf_depth, random_state=42)
    sub_rf.fit(X_train, y_train)
    rf_staged_acc.append(accuracy_score(y_test, sub_rf.predict(X_test)))

df_stages = pd.DataFrame({
    "Estimators": list(range(1, n_estimators + 1)) * 2,
    "Accuracy": ada_staged_acc + rf_staged_acc,
    "Algorithm": ["AdaBoost (Sequential)"] * n_estimators + ["Random Forest (Parallel)"] * n_estimators
})

fig2, ax3 = plt.subplots(figsize=(14, 4))
sns.lineplot(
    data=df_stages, x="Estimators", y="Accuracy", hue="Algorithm",
    palette={"AdaBoost (Sequential)": "crimson", "Random Forest (Parallel)": "royalblue"},
    linewidth=2.5, ax=ax3
)
ax3.set_title("Test Accuracy as More Trees/Stumps Are Added", fontweight="bold")
ax3.set_ylabel("Accuracy")
ax3.set_xlabel("Number of Estimators")
plt.tight_layout()
st.pyplot(fig2)

# ----------------- VISUALIZATION 3 & 4: MATRICES & PROBABILITY DENSITY -----------------
st.subheader("3. Error Analysis & Confidence Density")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Side-by-Side Confusion Matrices**")
    cm_ada = confusion_matrix(y_test, ada_preds)
    cm_rf = confusion_matrix(y_test, rf_preds)

    fig3, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.heatmap(cm_ada, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm1)
    ax_cm1.set_title("AdaBoost Confusion Matrix")
    ax_cm1.set_xlabel("Predicted")
    ax_cm1.set_ylabel("Actual")

    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_cm2)
    ax_cm2.set_title("Random Forest Confusion Matrix")
    ax_cm2.set_xlabel("Predicted")
    ax_cm2.set_ylabel("Actual")

    plt.tight_layout()
    st.pyplot(fig3)

with col_g2:
    st.markdown("**Confidence Distribution: $P(Class = 1)$**")
    df_probs = pd.DataFrame({
        "Confidence": np.concatenate([ada_probs, rf_probs]),
        "Model": ["AdaBoost"] * len(ada_probs) + ["Random Forest"] * len(rf_probs)
    })

    fig4, ax_kde = plt.subplots(figsize=(6, 3.8))
    sns.kdeplot(
        data=df_probs, x="Confidence", hue="Model",
        palette={"AdaBoost": "crimson", "Random Forest": "royalblue"},
        fill=True, common_norm=False, alpha=0.3, ax=ax_kde
    )
    ax_kde.axvline(0.5, color="black", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    ax_kde.set_title("Predicted Class Probability Distribution")
    ax_kde.set_xlabel("Predicted Probability of Class 1")
    ax_kde.legend(loc="upper right")
    plt.tight_layout()
    st.pyplot(fig4)

st.divider()

# ----------------- INTERACTIVE SINGLE POINT TESTING -----------------
st.subheader("4. Real-Time Coordinate Classifier")
c_x, c_y, c_btn = st.columns([2, 2, 2])
pt_x = c_x.number_input("Input Coordinate X", value=float(np.mean(X[:, 0])), format="%.2f")
pt_y = c_y.number_input("Input Coordinate Y", value=float(np.mean(X[:, 1])), format="%.2f")

with c_btn:
    st.write(" ")
    st.write(" ")
    evaluate = st.button("Evaluate Coordinate", type="primary")

if evaluate:
    sample = np.array([[pt_x, pt_y]])
    ada_res = ada_clf.predict(sample)[0]
    ada_cf = ada_clf.predict_proba(sample)[0][ada_res]

    rf_res = rf_clf.predict(sample)[0]
    rf_cf = rf_clf.predict_proba(sample)[0][rf_res]

    r1, r2 = st.columns(2)
    r1.info(f"**AdaBoost:** Class `{ada_res}` ({ada_cf * 100:.1f}% Confidence)")
    r2.success(f"**Random Forest:** Class `{rf_res}` ({rf_cf * 100:.1f}% Confidence)")
