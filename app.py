import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Credit Risk & Loan Default Engine", layout="wide")

# 1. Dataset Generation (FinTech Credit Data)
@st.cache_data
def generate_credit_data(n_samples=1000):
    np.random.seed(42)
    income = np.random.normal(55000, 18000, n_samples).clip(15000, 150000)
    credit_score = np.random.normal(650, 70, n_samples).clip(300, 850)
    loan_amount = np.random.normal(20000, 10000, n_samples).clip(2000, 60000)
    dti_ratio = np.random.uniform(0.05, 0.60, n_samples)
    emp_length = np.random.randint(0, 25, n_samples)

    risk_score = (
        - 0.00003 * income
        - 0.008 * credit_score
        + 0.00005 * loan_amount
        + 4.0 * dti_ratio
        - 0.08 * emp_length
        + 2.2
    )
    prob_default = 1 / (1 + np.exp(-risk_score))
    default = (np.random.rand(n_samples) < prob_default).astype(int)

    return pd.DataFrame({
        "Annual_Income": np.round(income, 2),
        "Credit_Score": np.round(credit_score).astype(int),
        "Loan_Amount": np.round(loan_amount, 2),
        "DTI_Ratio": np.round(dti_ratio, 3),
        "Employment_Years": emp_length,
        "Default": default
    })

df = generate_credit_data()
features = ["Annual_Income", "Credit_Score", "Loan_Amount", "DTI_Ratio", "Employment_Years"]
X = df[features]
y = df["Default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Sidebar Controls
st.sidebar.title("⚙️ Ensemble Settings (Module 3)")
n_estimators = st.sidebar.slider("Number of Estimators (T)", min_value=5, max_value=200, value=50, step=5)
learning_rate = st.sidebar.slider("AdaBoost Learning Rate (η)", min_value=0.01, max_value=2.0, value=0.5, step=0.05)

# Train Models
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=42
)
ada.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=5,
    random_state=42
)
rf.fit(X_train, y_train)

ada_preds = ada.predict(X_test)
ada_probs = ada.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# 3. Main Dashboard
st.title("🏦 Credit Risk & Loan Default Assessment Studio")
st.markdown("Comparative study of **Boosting (AdaBoost with Decision Stumps)** vs. **Bagging (Random Forest)**.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("⚡ Boosting: AdaBoost (Decision Stumps)")
    st.metric("Test Accuracy", f"{accuracy_score(y_test, ada_preds)*100:.2f}%")
    st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, ada_probs):.3f}")

with col2:
    st.subheader("🌲 Bagging: Random Forest")
    st.metric("Test Accuracy", f"{accuracy_score(y_test, rf_preds)*100:.2f}%")
    st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, rf_probs):.3f}")

st.divider()

# 4. Feature Importance Comparison
st.subheader("📊 Feature Importance Comparison")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].barh(features, ada.feature_importances_, color="#f56c42")
axes[0].set_title("AdaBoost (Stumps) Feature Weights")
axes[1].barh(features, rf.feature_importances_, color="#4287f5")
axes[1].set_title("Random Forest Gini Importances")
plt.tight_layout()
st.pyplot(fig)

st.divider()

# 5. Live Assessment
st.subheader("📝 Live Applicant Assessment")
c1, c2, c3, c4, c5 = st.columns(5)
in_income = c1.number_input("Annual Income ($)", value=45000, step=5000)
in_score = c2.slider("Credit Score", 300, 850, 620)
in_loan = c3.number_input("Loan Amount ($)", value=25000, step=2000)
in_dti = c4.slider("DTI Ratio", 0.05, 0.60, 0.35, step=0.01)
in_emp = c5.slider("Years Employed", 0, 25, 4)

applicant_data = pd.DataFrame([{
    "Annual_Income": in_income,
    "Credit_Score": in_score,
    "Loan_Amount": in_loan,
    "DTI_Ratio": in_dti,
    "Employment_Years": in_emp
}])

if st.button("Evaluate Loan Application", type="primary"):
    ada_prob = ada.predict_proba(applicant_data)[0, 1]
    rf_prob = rf.predict_proba(applicant_data)[0, 1]

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"**AdaBoost Default Risk:** `{ada_prob*100:.1f}%`")
        if ada_prob > 0.5:
            st.error("❌ High Risk: Rejected")
        else:
            st.success("✅ Low Risk: Approved")

    with r2:
        st.markdown(f"**Random Forest Default Risk:** `{rf_prob*100:.1f}%`")
        if rf_prob > 0.5:
            st.error("❌ High Risk: Rejected")
        else:
            st.success("✅ Low Risk: Approved")
