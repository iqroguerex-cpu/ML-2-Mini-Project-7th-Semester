import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Dynamic Ensemble Arena",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM DYNAMIC CSS -----------------
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Dynamic Ensemble Learning Arena")
st.markdown("##### Real-Time Interactive Laboratory: **Boosting (AdaBoost)** vs. **Bagging (Random Forest)**")

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("🎛️ Simulation Controls")
    dataset_name = st.selectbox("Pattern Geometry", ["Two Moons", "Concentric Circles", "Gaussian Blobs"])
    n_samples = st.slider("Sample Population", 100, 800, 350, step=50)
    noise_level = st.slider("Environmental Noise (Overlap)", 0.05, 0.40, 0.18, step=0.01)

    st.markdown("---")
    st.header("⚙️ Ensemble Architecture")
    n_estimators = st.slider("Total Ensemble Estimators ($N$)", 5, 80, 30, step=5)
    adaboost_lr = st.slider("AdaBoost Learning Rate (η)", 0.1, 2.0, 1.0, step=0.1)
    rf_depth = st.slider("Random Forest Tree Depth", 1, 10, 4, step=1)

    st.markdown("---")
    resample_btn = st.button("🎲 Reseed & Regenerate Data", use_container_width=True)

# ----------------- DATASET GENERATION -----------------
seed = np.random.randint(1, 9999) if resample_btn else 42

@st.cache_data
def generate_data(name, n, noise, current_seed):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=current_seed)
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=current_seed)
    else:
        X, y = make_blobs(n_samples=n, centers=2, cluster_std=noise * 8, random_state=current_seed)
    return X, y

X, y = generate_data(dataset_name, n_samples, noise_level, seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------- MODEL INITIALIZATION & TRAINING -----------------
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

# ----------------- METRIC CARDS -----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("AdaBoost Accuracy", f"{ada_acc * 100:.1f}%", delta=f"{ada_f1:.3f} F1")
m2.metric("Random Forest Accuracy", f"{rf_acc * 100:.1f}%", delta=f"{rf_f1:.3f} F1")
lead_diff = (ada_acc - rf_acc) * 100
if lead_diff > 0:
    m3.metric("Competitive Margin", f"+{lead_diff:.1f}%", delta="AdaBoost Advantage")
elif lead_diff < 0:
    m3.metric("Competitive Margin", f"+{abs(lead_diff):.1f}%", delta="Random Forest Advantage", delta_color="inverse")
else:
    m3.metric("Competitive Margin", "0.0%", delta="Parity (Tie)")
m4.metric("Holdout Test Points", f"{len(X_test)} samples")

# ----------------- INTERACTIVE DYNAMIC INSPECTOR -----------------
st.markdown("### 1. Interactive Meshgrid: Decision Topography")
st.caption("Hover over regions to inspect continuous class decision boundaries and underlying data points.")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
grid_res = 120
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_ada = ada_clf.predict(grid_points).reshape(xx.shape)
Z_rf = rf_clf.predict(grid_points).reshape(xx.shape)

fig_mesh = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"<b>AdaBoost (Boosting)</b> — Acc: {ada_acc*100:.1f}%",
        f"<b>Random Forest (Bagging)</b> — Acc: {rf_acc*100:.1f}%"
    ),
    horizontal_spacing=0.08
)

# AdaBoost Surface & Scatter
fig_mesh.add_trace(
    go.Contour(
        x=np.linspace(x_min, x_max, grid_res),
        y=np.linspace(y_min, y_max, grid_res),
        z=Z_ada,
        colorscale="RdBu",
        opacity=0.35,
        showscale=False,
        hoverinfo="skip"
    ),
    row=1, col=1
)
for target, color, label in [(0, "#3b82f6", "Class 0"), (1, "#ef4444", "Class 1")]:
    mask = (y_test == target)
    fig_mesh.add_trace(
        go.Scatter(
            x=X_test[mask, 0], y=X_test[mask, 1],
            mode="markers",
            marker=dict(color=color, size=7, line=dict(width=1, color="black")),
            name=f"Ada: {label}",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

# Random Forest Surface & Scatter
fig_mesh.add_trace(
    go.Contour(
        x=np.linspace(x_min, x_max, grid_res),
        y=np.linspace(y_min, y_max, grid_res),
        z=Z_rf,
        colorscale="RdBu",
        opacity=0.35,
        showscale=False,
        hoverinfo="skip"
    ),
    row=1, col=2
)
for target, color, label in [(0, "#3b82f6", "Class 0"), (1, "#ef4444", "Class 1")]:
    mask = (y_test == target)
    fig_mesh.add_trace(
        go.Scatter(
            x=X_test[mask, 0], y=X_test[mask, 1],
            mode="markers",
            marker=dict(color=color, size=7, line=dict(width=1, color="black")),
            name=f"RF: {label}",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

fig_mesh.update_layout(height=480, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
st.plotly_chart(fig_mesh, use_container_width=True)

# ----------------- DYNAMIC SLIDER: STEP-THROUGH SIMULATION -----------------
st.markdown("### 2. Time-Travel: Ensemble Growth Simulator")
st.caption("Scrub through iterations to see how performance scales as base estimators join the committee.")

step_iter = st.slider("Examine Ensemble at Stage (Tree count)", 1, n_estimators, min(10, n_estimators), step=1)

# Progressive evaluation
ada_staged = list(ada_clf.staged_predict(X_test))[:step_iter]
ada_curve = [accuracy_score(y_test, p) for p in ada_staged]

rf_sub = RandomForestClassifier(n_estimators=step_iter, max_depth=rf_depth, random_state=42).fit(X_train, y_train)
rf_sub_acc = accuracy_score(y_test, rf_sub.predict(X_test))

col_step1, col_step2 = st.columns([3, 1])

with col_step1:
    fig_trace = go.Figure()
    fig_trace.add_trace(go.Scatter(
        x=list(range(1, step_iter + 1)), y=ada_curve,
        mode="lines+markers", name="AdaBoost Staged Accuracy",
        line=dict(color="#ef4444", width=3)
    ))
    fig_trace.add_hline(
        y=rf_sub_acc, line_dash="dash", line_color="#3b82f6",
        annotation_text=f"RF Accuracy @ {step_iter} trees: {rf_sub_acc*100:.1f}%",
        annotation_position="bottom right"
    )
    fig_trace.update_layout(
        title=f"Accuracy Trajectory up to Estimator {step_iter}",
        xaxis_title="Estimator Iteration",
        yaxis_title="Validation Accuracy",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_trace, use_container_width=True)

with col_step2:
    st.markdown(f"**Stage {step_iter} Status**")
    st.write(f"AdaBoost Iteration Acc: **{ada_curve[-1]*100:.1f}%**")
    st.write(f"Random Forest Snapshot Acc: **{rf_sub_acc*100:.1f}%**")
    if hasattr(ada_clf, "estimator_weights_"):
        st.write(f"Latest Stump Weight ($\\alpha$): `{ada_clf.estimator_weights_[step_iter-1]:.3f}`")
    st.progress(float(ada_curve[-1]))

# ----------------- PROBABILITY DENSITY & CONFUSION MATRIX -----------------
st.markdown("### 3. Calibration & Confidence Inspector")
col_c1, col_c2 = st.columns(2)

with col_c1:
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=ada_probs, nbinsx=25, name="AdaBoost $P(Class=1)$",
        opacity=0.6, marker_color="#ef4444"
    ))
    fig_conf.add_trace(go.Histogram(
        x=rf_probs, nbinsx=25, name="Random Forest $P(Class=1)$",
        opacity=0.6, marker_color="#3b82f6"
    ))
    fig_conf.update_layout(
        barmode="overlay",
        title="Prediction Confidence Density Distribution",
        xaxis_title="Posterior Probability $P(y=1\vert{}x)$",
        yaxis_title="Frequency Count",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_conf, use_container_width=True)

with col_c2:
    cm_ada = confusion_matrix(y_test, ada_preds)
    cm_rf = confusion_matrix(y_test, rf_preds)
    fig_cm = make_subplots(rows=1, cols=2, subplot_titles=("AdaBoost Matrix", "Random Forest Matrix"))
    fig_cm.add_trace(go.Heatmap(z=cm_ada, text=cm_ada, texttemplate="%{text}", colorscale="Reds", showscale=False), row=1, col=1)
    fig_cm.add_trace(go.Heatmap(z=cm_rf, text=cm_rf, texttemplate="%{text}", colorscale="Blues", showscale=False), row=1, col=2)
    fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

# ----------------- REAL-TIME LIVE PROBE DIAGNOSTIC -----------------
st.markdown("---")
st.subheader("🎯 Real-Time Probe Diagnostic")
st.caption("Enter arbitrary values or generate a random sample to test decision boundary inference.")

probe_c1, probe_c2, probe_c3, probe_c4 = st.columns([2, 2, 2, 2])

with probe_c1:
    probe_x = st.number_input("Probe Feature 1", value=float(np.mean(X[:, 0])), format="%.2f")
with probe_c2:
    probe_y = st.number_input("Probe Feature 2", value=float(np.mean(X[:, 1])), format="%.2f")
with probe_c3:
    st.write(" ")
    st.write(" ")
    eval_btn = st.button("⚡ Evaluate Coordinates", type="primary", use_container_width=True)
with probe_c4:
    st.write(" ")
    st.write(" ")
    rand_btn = st.button("🎲 Random Probe", use_container_width=True)

if rand_btn:
    rand_idx = np.random.randint(0, len(X_test))
    probe_x = float(X_test[rand_idx, 0])
    probe_y = float(X_test[rand_idx, 1])
    eval_btn = True

if eval_btn:
    sample = np.array([[probe_x, probe_y]])
    a_pred = ada_clf.predict(sample)[0]
    a_prob = ada_clf.predict_proba(sample)[0]

    r_pred = rf_clf.predict(sample)[0]
    r_prob = rf_clf.predict_proba(sample)[0]

    out1, out2 = st.columns(2)
    with out1:
        st.markdown(f"**AdaBoost Prediction:** Class `{a_pred}`")
        st.caption(f"Confidence: Class 0 ({a_prob[0]*100:.1f}%) | Class 1 ({a_prob[1]*100:.1f}%)")
        st.progress(float(a_prob[1]))

    with out2:
        st.markdown(f"**Random Forest Prediction:** Class `{r_pred}`")
        st.caption(f"Confidence: Class 0 ({r_prob[0]*100:.1f}%) | Class 1 ({r_prob[1]*100:.1f}%)")
        st.progress(float(r_prob[1]))
