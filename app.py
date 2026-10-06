import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Quantitative HMM & MCMC Regime Engine", layout="wide")

st.title("Financial Regime Switcher: Baum-Welch, Viterbi & MCMC Calibration")
st.caption("Module 4 (MCMC Sampling) & Module 5 (HMM, Baum-Welch, Viterbi Algorithm)")

# ----------------- SIDEBAR PARAMETERS -----------------
st.sidebar.header("1. Market Simulation Config")
n_obs = st.sidebar.slider("Trading Days (T)", 100, 800, 300, step=50)
vol_multiplier = st.sidebar.slider("Regime Volatility Ratio", 1.5, 6.0, 3.5, step=0.5)

st.sidebar.header("2. Baum-Welch (EM) Settings")
em_iters = st.sidebar.slider("Baum-Welch Iterations", 5, 50, 20, step=5)
tolerance = 1e-4

st.sidebar.header("3. Module 4: MCMC Metropolis-Hastings")
n_mcmc = st.sidebar.slider("MCMC Trace Length", 500, 4000, 1500, step=250)
proposal_std = st.sidebar.slider("Gaussian Proposal Variance (std)", 0.01, 0.5, 0.08, step=0.01)

# ----------------- SYNTHETIC REGIME GENERATOR -----------------
@st.cache_data
def generate_market_data(T, vol_ratio):
    np.random.seed(42)
    # 2 Hidden States: 0 = Bull/Calm (Low vol, positive drift), 1 = Bear/Stress (High vol, negative drift)
    true_A = np.array([[0.95, 0.05], [0.10, 0.90]])
    true_mu = [0.001, -0.002]
    true_sigma = [0.01, 0.01 * vol_ratio]

    states = np.zeros(T, dtype=int)
    returns = np.zeros(T)
    states[0] = 0

    for t in range(1, T):
        states[t] = np.random.choice([0, 1], p=true_A[states[t - 1]])
    
    for t in range(T):
        returns[t] = np.random.normal(true_mu[states[t]], true_sigma[states[t]])

    prices = 100 * np.exp(np.cumsum(returns))
    return returns, prices, states, true_mu, true_sigma, true_A

returns, prices, true_states, true_mu, true_sigma, true_A = generate_market_data(n_obs, vol_multiplier)

# ----------------- MODULE 5: HMM (BAUM-WELCH & VITERBI) -----------------
class GaussianHMM:
    def __init__(self, n_states=2):
        self.K = n_states
        self.pi = np.array([0.5, 0.5])
        self.A = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.mu = np.array([0.002, -0.002])
        self.sigma = np.array([0.01, 0.03])

    def emission_prob(self, x):
        T = len(x)
        B = np.zeros((self.K, T))
        for k in range(self.K):
            B[k, :] = norm.pdf(x, self.mu[k], np.maximum(self.sigma[k], 1e-4)) + 1e-12
        return B

    def forward_backward(self, x):
        T = len(x)
        B = self.emission_prob(x)
        alpha = np.zeros((self.K, T))
        beta = np.zeros((self.K, T))
        scale = np.zeros(T)

        # Forward
        alpha[:, 0] = self.pi * B[:, 0]
        scale[0] = np.sum(alpha[:, 0]) + 1e-12
        alpha[:, 0] /= scale[0]

        for t in range(1, T):
            alpha[:, t] = (self.A.T @ alpha[:, t - 1]) * B[:, t]
            scale[t] = np.sum(alpha[:, t]) + 1e-12
            alpha[:, t] /= scale[t]

        # Backward
        beta[:, -1] = 1.0 / scale[-1]
        for t in range(T - 2, -1, -1):
            beta[:, t] = (self.A @ (beta[:, t + 1] * B[:, t + 1])) / scale[t]

        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0, keepdims=True)

        xi = np.zeros((self.K, self.K, T - 1))
        for t in range(T - 1):
            denom = np.sum(alpha[:, t, None] * self.A * B[:, t + 1] * beta[:, t + 1]) + 1e-12
            xi[:, :, t] = (alpha[:, t, None] * self.A * B[:, t + 1] * beta[:, t + 1]) / denom

        return alpha, beta, gamma, xi

    def fit_baum_welch(self, x, iterations):
        T = len(x)
        for _ in range(iterations):
            alpha, beta, gamma, xi = self.forward_backward(x)
            self.pi = gamma[:, 0]
            self.A = np.sum(xi, axis=2) / (np.sum(gamma[:, :-1], axis=1)[:, None] + 1e-12)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            for k in range(self.K):
                gamma_k = gamma[k, :]
                w_sum = np.sum(gamma_k) + 1e-12
                self.mu[k] = np.sum(gamma_k * x) / w_sum
                self.sigma[k] = np.sqrt(np.sum(gamma_k * (x - self.mu[k])**2) / w_sum)

    def viterbi(self, x):
        T = len(x)
        B = self.emission_prob(x)
        delta = np.zeros((self.K, T))
        psi = np.zeros((self.K, T), dtype=int)

        delta[:, 0] = np.log(self.pi + 1e-12) + np.log(B[:, 0])
        for t in range(1, T):
            for j in range(self.K):
                seq_probs = delta[:, t - 1] + np.log(self.A[:, j] + 1e-12)
                psi[j, t] = np.argmax(seq_probs)
                delta[j, t] = np.max(seq_probs) + np.log(B[j, t])

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(delta[:, -1])
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[best_path[t + 1], t + 1]
        return best_path

hmm = GaussianHMM(n_states=2)
hmm.fit_baum_welch(returns, em_iters)
inferred_states = hmm.viterbi(returns)

# Correct potential label flipping based on volatility scale
if hmm.sigma[0] > hmm.sigma[1]:
    inferred_states = 1 - inferred_states

# ----------------- MODULE 4: MCMC METROPOLIS-HASTINGS SAMPLER -----------------
def run_mcmc(data, states, n_samples, prop_std):
    # Sample posterior distribution of Bear-State volatility: P(sigma_bear | observations)
    bear_data = data[states == 1]
    if len(bear_data) < 5:
        bear_data = data  # Fallback

    samples = np.zeros(n_samples)
    curr_sigma = np.std(bear_data)
    samples[0] = curr_sigma

    def log_target(s):
        if s <= 0:
            return -np.inf
        # Gaussian log-likelihood with Half-Cauchy prior for scale
        ll = np.sum(norm.logpdf(bear_data, loc=np.mean(bear_data), scale=s))
        l_prior = -np.log(1.0 + (s / 0.05)**2)
        return ll + l_prior

    curr_ll = log_target(curr_sigma)
    accepted = 0

    for i in range(1, n_samples):
        candidate = curr_sigma + np.random.normal(0, prop_std)
        cand_ll = log_target(candidate)
        alpha = np.exp(cand_ll - curr_ll)

        if np.random.rand() < alpha:
            curr_sigma = candidate
            curr_ll = cand_ll
            accepted += 1
        samples[i] = curr_sigma

    return samples[int(n_samples * 0.2):], accepted / n_samples

mcmc_samples, acceptance_rate = run_mcmc(returns, inferred_states, n_mcmc, proposal_std)

# ----------------- UI / PLOTS -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Learned Low-Vol (Bull)", f"{np.min(hmm.sigma):.4f}")
col2.metric("Learned High-Vol (Bear)", f"{np.max(hmm.sigma):.4f}")
col3.metric("MCMC Posterior Mean (Bear)", f"{np.mean(mcmc_samples):.4f}")
col4.metric("MCMC Proposal Acceptance", f"{acceptance_rate * 100:.1f}%")

st.subheader("1. Inferred Hidden Regimes (Viterbi Path vs Asset Price)")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(prices, color="black", label="Asset Price (Synthetic Index)")
bear_mask = (inferred_states == 1)
ax1.fill_between(range(n_obs), np.min(prices), np.max(prices), where=bear_mask, color='crimson', alpha=0.25, label="High-Risk Bear Regime")
ax1.set_ylabel("Price")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(returns, color="royalblue", alpha=0.7, label="Log Returns")
ax2.set_ylabel("Returns")
ax2.set_xlabel("Trading Days")
ax2.grid(True, linestyle=":", alpha=0.6)
st.pyplot(fig1)

st.subheader("2. Markov Chain Monte Carlo: Posterior Volatility Distribution")
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4))

ax3.plot(mcmc_samples, color="teal", alpha=0.8)
ax3.set_title("MCMC Trace Plot (Convergence Verification)")
ax3.set_xlabel("Iteration (Post-Burn-in)")
ax3.set_ylabel(r"Parameter $\sigma_{bear}$")
ax3.grid(True, linestyle=":", alpha=0.6)

ax4.hist(mcmc_samples, bins=30, density=True, color="teal", alpha=0.6, edgecolor="black")
ax4.axvline(np.mean(mcmc_samples), color="red", linestyle="--", label=f"Posterior Mean: {np.mean(mcmc_samples):.4f}")
ax4.set_title(r"Posterior Density $P(\sigma_{bear} | D)$")
ax4.set_xlabel(r"Estimated $\sigma_{bear}$")
ax4.legend()
ax4.grid(True, linestyle=":", alpha=0.6)
st.pyplot(fig2)
