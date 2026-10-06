import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Kohonen SOM Telemetry Engine", layout="wide")

st.title("Self-Organizing Map (SOM) Topological Telemetry Engine")
st.caption("Module 4: Vector Quantisation, The SOM Algorithm, Neighbourhood Connections & Boundary Conditions")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("1. SOM Lattice Hyperparameters")
grid_size = st.sidebar.slider("Grid Dimension (N x N)", 6, 16, 10, step=2)
epochs = st.sidebar.slider("Training Epochs", 20, 200, 80, step=10)
initial_lr = st.sidebar.slider("Initial Learning Rate (η0)", 0.1, 1.0, 0.5, step=0.05)
initial_sigma = st.sidebar.slider("Initial Neighborhood Radius (σ0)", 1.0, float(grid_size / 2), float(grid_size / 2), step=0.5)

st.sidebar.header("2. Telemetry Simulation")
n_samples = st.sidebar.slider("Normal Telemetry Samples", 200, 1000, 400, step=50)
anomaly_ratio = st.sidebar.slider("Injected Anomaly Ratio", 0.02, 0.20, 0.08, step=0.01)

# ----------------- DATASET GENERATOR -----------------
# 4 Real-World Sensor Telemetry Features:
# [CPU Load (%), Core Temp (°C), Power Draw (W), Fan RPM (x1000)]
@st.cache_data
def generate_telemetry_data(n_normal, anom_ratio):
    np.random.seed(42)
    # Cluster 1: Idle state
    idle = np.random.multivariate_normal(
        mean=[20, 42, 60, 1.8],
        cov=np.diag([15, 6, 25, 0.04]),
        size=int(n_normal * 0.45)
    )
    # Cluster 2: Heavy compute state
    compute = np.random.multivariate_normal(
        mean=[85, 78, 190, 4.2],
        cov=np.diag([20, 8, 40, 0.06]),
        size=int(n_normal * 0.55)
    )
    normal_data = np.vstack([idle, compute])

    # Anomalies: Thermal throttling, fan failure, voltage spikes
    n_anomalies = int(n_normal * anom_ratio)
    anom_cpu = np.random.uniform(90, 100, n_anomalies)
    anom_temp = np.random.uniform(95, 110, n_anomalies)  # Critical heat
    anom_pwr = np.random.uniform(220, 260, n_anomalies)
    anom_fan = np.random.uniform(0.5, 1.2, n_anomalies)   # Dead fan
    anomalies = np.column_stack([anom_cpu, anom_temp, anom_pwr, anom_fan])

    data = np.vstack([normal_data, anomalies])
    labels = np.array([0] * len(normal_data) + [1] * len(anomalies))

    # Min-Max Normalization (essential for Kohonen SOM)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-12)

    return normalized, labels, min_vals, max_vals

norm_data, true_labels, min_vals, max_vals = generate_telemetry_data(n_samples, anomaly_ratio)
feature_names = ["CPU Load (%)", "Core Temp (°C)", "Power Draw (W)", "Fan RPM (k)"]

# ----------------- KOHONEN SOM FROM SCRATCH -----------------
class KohonenSOM:
    def __init__(self, height, width, dim, lr=0.5, sigma=None):
        self.height = height
        self.width = width
        self.dim = dim
        self.lr0 = lr
        self.sigma0 = sigma if sigma else max(height, width) / 2.0
        # Initialize weight lattice randomly in normalized space
        np.random.seed(42)
        self.weights = np.random.uniform(0.1, 0.9, (height, width, dim))

        # Precompute 2D lattice coordinate grid for fast distance calculation
        self.grid_coords = np.zeros((height, width, 2))
        for r in range(height):
            for c in range(width):
                self.grid_coords[r, c] = [r, c]

    def find_bmu(self, x):
        """Find Best Matching Unit (BMU) for input vector x."""
        # Euclidean distance along feature dimension: shape (height, width)
        dists = np.linalg.norm(self.weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dists), (self.height, self.width))
        return bmu_idx, dists[bmu_idx]

    def train(self, data, max_epochs):
        n_samples = len(data)
        time_constant = max_epochs / np.log(self.sigma0 + 1e-12)
        error_progression = []

        progress_bar = st.progress(0, text="Training SOM Lattice...")

        for epoch in range(max_epochs):
            # Dynamic time-decayed learning rate and neighborhood radius
            eta = self.lr0 * np.exp(-epoch / max_epochs)
            sigma = self.sigma0 * np.exp(-epoch / time_constant)
            sigma_sq_2 = 2.0 * (sigma ** 2) + 1e-12

            # Shuffle samples each epoch
            indices = np.random.permutation(n_samples)
            epoch_quant_error = 0.0

            for i in indices:
                x = data[i]
                bmu, min_dist = self.find_bmu(x)
                epoch_quant_error += min_dist

                # Topological distance squared to BMU on lattice: ||r_i - r_bmu||^2
                dist_to_bmu = np.sum((self.grid_coords - np.array(bmu)) ** 2, axis=2)

                # Gaussian neighborhood function: h_ci(t) = exp(-dist^2 / (2 * sigma^2))
                h = np.exp(-dist_to_bmu / sigma_sq_2)

                # Vector Quantisation Weight Update: w_i = w_i + eta * h_ci * (x - w_i)
                self.weights += eta * h[:, :, np.newaxis] * (x - self.weights)

            error_progression.append(epoch_quant_error / n_samples)
            progress_bar.progress((epoch + 1) / max_epochs, text=f"Epoch {epoch + 1}/{max_epochs} complete")

        progress_bar.empty()
        return error_progression

    def compute_umatrix(self):
        """Computes Unified Distance Matrix (U-Matrix) showing topological separation."""
        u_matrix = np.zeros((self.height, self.width))
        for r in range(self.height):
            for c in range(self.width):
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbors.append(np.linalg.norm(self.weights[r, c] - self.weights[nr, nc]))
                u_matrix[r, c] = np.mean(neighbors) if neighbors else 0.0
        return u_matrix

# Run training
som = KohonenSOM(grid_size, grid_size, dim=4, lr=initial_lr, sigma=initial_sigma)
errors = som.train(norm_data, epochs)
u_mat = som.compute_umatrix()

# Calculate quantization error for all data points
bmu_distances = np.array([som.find_bmu(x)[1] for x in norm_data])
# Threshold for anomaly detection: mean + 2 * std
anomaly_threshold = np.mean(bmu_distances) + 2.0 * np.std(bmu_distances)
flagged_anomalies = bmu_distances > anomaly_threshold

# ----------------- UI METRICS -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Lattice Topology", f"{grid_size} x {grid_size} Neurons")
col2.metric("Final Quantization Error", f"{errors[-1]:.4f}")
col3.metric("Anomaly Cutoff Threshold", f"{anomaly_threshold:.4f}")
col4.metric("Telemetry Anomalies Flagged", f"{np.sum(flagged_anomalies)} / {len(norm_data)}")

st.divider()

# ----------------- VISUALIZATIONS -----------------
tab1, tab2 = st.tabs(["Topological U-Matrix & Feature Weights", "Quantization Error & Training Convergence"])

with tab1:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. U-Matrix with data projection
    im0 = axes[0].imshow(u_mat, cmap="bone_r", origin="lower")
    axes[0].set_title("Unified Distance Matrix (U-Matrix)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Project normal vs anomaly BMU coordinates onto U-Matrix
    for i, x in enumerate(norm_data):
        bmu, _ = som.find_bmu(x)
        if true_labels[i] == 1:
            axes[0].scatter(bmu[1], bmu[0], color="red", s=40, marker="x", label="Anomaly" if i == len(norm_data) - 1 else "")
        else:
            axes[0].scatter(bmu[1], bmu[0], color="cyan", s=10, alpha=0.3)
    axes[0].set_xlabel("Lattice X")
    axes[0].set_ylabel("Lattice Y")

    # 2. Temperature Weight Component Plane
    im1 = axes[1].imshow(som.weights[:, :, 1], cmap="inferno", origin="lower")
    axes[1].set_title("Component Plane: Core Temp (°C)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xlabel("Lattice X")

    # 3. Fan RPM Weight Component Plane
    im2 = axes[2].imshow(som.weights[:, :, 3], cmap="viridis", origin="lower")
    axes[2].set_title("Component Plane: Fan RPM (k)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_xlabel("Lattice X")

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig2, (ax_err, ax_quant) = plt.subplots(1, 2, figsize=(14, 4))

    ax_err.plot(errors, color="royalblue", lw=2)
    ax_err.set_title("Quantization Error vs Epoch (Convergence)")
    ax_err.set_xlabel("Epoch")
    ax_err.set_ylabel("Mean Reconstruction Error")
    ax_err.grid(True, linestyle="--", alpha=0.6)

    # Anomaly Quantization Histogram
    ax_quant.hist(bmu_distances[true_labels == 0], bins=25, alpha=0.7, color="teal", label="Normal Telemetry")
    ax_quant.hist(bmu_distances[true_labels == 1], bins=15, alpha=0.8, color="crimson", label="Ground Truth Anomalies")
    ax_quant.axvline(anomaly_threshold, color="black", linestyle="--", lw=2, label=f"Cutoff ({anomaly_threshold:.3f})")
    ax_quant.set_title("BMU Distance Distribution (Quantization Loss)")
    ax_quant.set_xlabel("Distance to Nearest Prototype Neuron")
    ax_quant.legend()
    ax_quant.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig2)

# ----------------- REAL-WORLD INFERENCE INSPECTOR -----------------
st.subheader("3. Live Telemetry Packet Anomaly Diagnostic")
st.markdown("Inject an unseen multi-sensor telemetry stream to evaluate prototype distance against the trained lattice.")

col_a, col_b, col_c, col_d = st.columns(4)
input_cpu = col_a.slider("Input CPU Load (%)", 0.0, 100.0, 95.0)
input_temp = col_b.slider("Input Core Temp (°C)", 30.0, 115.0, 104.0)
input_pwr = col_c.slider("Input Power Draw (W)", 40.0, 280.0, 240.0)
input_fan = col_d.slider("Input Fan RPM (k)", 0.3, 5.0, 0.8)

if st.button("Evaluate Telemetry Packet", type="primary"):
    raw_packet = np.array([input_cpu, input_temp, input_pwr, input_fan])
    norm_packet = (raw_packet - min_vals) / (max_vals - min_vals + 1e-12)

    bmu_pos, quant_err = som.find_bmu(norm_packet)

    st.write(f"**Activated Best Matching Unit (BMU):** Grid Coordinate `(Row {bmu_pos[0]}, Col {bmu_pos[1]})`")
    st.write(f"**Quantization Distance:** `{quant_err:.4f}` (Anomaly Threshold: `{anomaly_threshold:.4f}`)")

    if quant_err > anomaly_threshold:
        st.error(f"🚨 CRITICAL OUTLIER DETECTED: Reconstruction error ({quant_err:.4f}) exceeds lattice threshold. Hardware failure or thermal runaway signature.")
    else:
        st.success(f"✅ NORMAL TELEMETRY: Sample successfully mapped to stable operational prototype cluster.")
