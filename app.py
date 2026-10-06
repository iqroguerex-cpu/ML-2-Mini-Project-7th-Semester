import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="FOIL Inductive Logic Engine", layout="wide")

st.title("First-Order Inductive Learner (FOIL) Rule Synthesis")
st.caption("Module 2: Learning First-Order Rules & Sequential Covering Algorithms")

# ----------------- DOMAIN RELATIONS & PREDICATES -----------------
# Target Concept: UnauthorizedAccess(User, Resource)
# Background Relations: Role, ResourceTier, Location, Protocol

DEFAULT_DATA = [
    # User, Role, Location, Resource, Tier, Protocol, Label
    ("alice", "intern", "office", "server_prod", "critical", "ssh", "+"),
    ("bob", "intern", "remote", "server_prod", "critical", "http", "+"),
    ("charlie", "engineer", "remote", "server_prod", "critical", "ssh", "-"),
    ("david", "intern", "office", "server_dev", "standard", "http", "-"),
    ("eve", "contractor", "remote", "server_prod", "critical", "ssh", "+"),
    ("frank", "engineer", "office", "server_prod", "critical", "ssh", "-"),
    ("grace", "contractor", "office", "server_dev", "standard", "http", "-"),
    ("heidi", "intern", "remote", "server_dev", "standard", "ssh", "+"),
    ("ivan", "contractor", "remote", "server_dev", "standard", "http", "+"),
    ("judy", "engineer", "remote", "server_dev", "standard", "ssh", "-"),
]

COLUMNS = ["User", "Role", "Location", "Resource", "Tier", "Protocol", "Label"]

st.sidebar.header("1. Relational Knowledge Base")
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(DEFAULT_DATA, columns=COLUMNS)

with st.sidebar.expander("Add Relational Fact", expanded=False):
    with st.form("add_fact"):
        u = st.text_input("User", "mallory")
        r = st.selectbox("Role", ["intern", "engineer", "contractor"])
        loc = st.selectbox("Location", ["office", "remote"])
        res = st.selectbox("Resource", ["server_prod", "server_dev"])
        tier = st.selectbox("Tier", ["critical", "standard"])
        proto = st.selectbox("Protocol", ["ssh", "http"])
        lbl = st.selectbox("Label (+ for Flagged, - for Allowed)", ["+", "-"])
        if st.form_submit_button("Append Fact"):
            new_row = pd.DataFrame([[u, r, loc, res, tier, proto, lbl]], columns=COLUMNS)
            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
            st.rerun()

# ----------------- FOIL ALGORITHM IMPLEMENTATION -----------------
def foil_gain(p0, n0, p1, n1):
    """Calculates FOIL Information Gain: Gain = t * (log2(p1/(p1+n1)) - log2(p0/(p0+n0)))"""
    if p1 == 0:
        return -float("inf")
    info_old = math.log2(p0 / (p0 + n0)) if (p0 + n0) > 0 else 0
    info_new = math.log2(p1 / (p1 + n1)) if (p1 + n1) > 0 else 0
    gain = p1 * (info_new - info_old)
    return gain

def run_foil(df):
    """Sequential covering algorithm searching for first-order body literals."""
    feature_cols = ["Role", "Location", "Tier", "Protocol"]
    pos_df = df[df["Label"] == "+"].copy()
    neg_df = df[df["Label"] == "-"].copy()

    learned_rules = []
    rule_metrics = []

    # Sequential Covering: Loop until all positive examples are covered
    remaining_pos = pos_df.copy()

    while len(remaining_pos) > 0:
        current_rule_literals = []
        current_pos = remaining_pos.copy()
        current_neg = neg_df.copy()

        # Build single clause body
        while len(current_neg) > 0:
            best_literal = None
            best_gain = -float("inf")
            best_pos_split = None
            best_neg_split = None

            p0 = len(current_pos)
            n0 = len(current_neg)

            # Evaluate candidate literals Predicate(Var, Value)
            for col in feature_cols:
                values = df[col].unique()
                for val in values:
                    candidate_literal = f"{col}(X, '{val}')"
                    if candidate_literal in current_rule_literals:
                        continue

                    cand_pos = current_pos[current_pos[col] == val]
                    cand_neg = current_neg[current_neg[col] == val]

                    p1 = len(cand_pos)
                    n1 = len(cand_neg)

                    if p1 == 0:
                        continue

                    gain = foil_gain(p0, n0, p1, n1)

                    if gain > best_gain:
                        best_gain = gain
                        best_literal = candidate_literal
                        best_pos_split = cand_pos
                        best_neg_split = cand_neg

            if best_literal is None or best_gain <= 0:
                # If no positive gain possible, break to avoid infinite loop
                break

            current_rule_literals.append(best_literal)
            current_pos = best_pos_split
            current_neg = best_neg_split

            if len(current_neg) == 0:
                break

        if not current_rule_literals:
            break

        rule_str = "UnauthorizedAccess(X, Y) :- " + ", ".join(current_rule_literals)
        covered_indices = current_pos.index
        learned_rules.append(rule_str)
        rule_metrics.append((len(current_pos), len(current_neg)))

        # Remove covered positive examples (Sequential Covering)
        remaining_pos = remaining_pos.drop(index=covered_indices, errors="ignore")

    return learned_rules, rule_metrics

import re

# ----------------- INFERENCE ENGINE -----------------
def evaluate_rules(rules, query_dict):
    """Evaluates whether an unseen access request satisfies any learned Horn clause."""
    fired_rules = []
    # Regex to robustly capture Predicate and Value from: Predicate(Var, 'Value') or Predicate(Var, Value)
    pattern = re.compile(r"(\w+)\s*\(\s*\w+\s*,\s*['\"]?([^'\")]+)['\"]?\s*\)")

    for r in rules:
        if ":- " not in r:
            continue
        body = r.split(":- ")[1]
        literals = [lit.strip() for lit in body.split(",")]
        match = True

        for lit in literals:
            m = pattern.search(lit)
            if not m:
                continue
            col, val = m.group(1), m.group(2)
            if query_dict.get(col) != val:
                match = False
                break

        if match and len(literals) > 0:
            fired_rules.append(r)

    return fired_rules

# ----------------- UI / WORKSPACE -----------------
st.subheader("1. Active Knowledge Base Facts")
st.dataframe(st.session_state.data, use_container_width=True)

st.subheader("2. Inductive Logic Rule Synthesis")
rules, metrics = run_foil(st.session_state.data)

if rules:
    for idx, (rule, (pos_cov, neg_cov)) in enumerate(zip(rules, metrics)):
        st.success(f"**Clause {idx + 1}:** `{rule}`  \n*(Covers {pos_cov} Positive Facts, {neg_cov} Negative Facts)*")
else:
    st.warning("No clauses could be induced. Ensure balanced positive/negative examples.")

st.divider()

# ----------------- REAL-WORLD TEST BENCH -----------------
st.subheader("3. Zero-Trust Access Request Inspector")
col1, col2, col3, col4 = st.columns(4)
test_role = col1.selectbox("Query Role", ["intern", "engineer", "contractor"])
test_loc = col2.selectbox("Query Location", ["office", "remote"])
test_tier = col3.selectbox("Query Tier", ["critical", "standard"])
test_proto = col4.selectbox("Query Protocol", ["ssh", "http"])

query_sample = {
    "Role": test_role,
    "Location": test_loc,
    "Tier": test_tier,
    "Protocol": test_proto
}

if st.button("Evaluate Access Security", type="primary"):
    matched = evaluate_rules(rules, query_sample)
    if matched:
        st.error("🚨 ACCESS DENIED: Triggered Security Violation Horn Clause(s):")
        for m in matched:
            st.code(m, language="prolog")
    else:
        st.success("✅ ACCESS GRANTED: Request satisfies safe operational predicates.")
