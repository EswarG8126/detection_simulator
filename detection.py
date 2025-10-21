# Prototype ML system for cloud anomaly detection + root-cause classification + automated mitigation
# Fixed version addressing test case failures
# This will:
# 1. Simulate telemetry time-series data with injected anomalies of different types.
# 2. Train an IsolationForest to detect anomalies (unsupervised).
# 3. Train a RandomForest classifier on labeled anomalies to predict root cause.
# 4. Demonstrate a simple automated mitigation executor (simulated).
# 5. Plot results (matplotlib required by tool instructions) and show a sample dataframe.

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import datetime
import random
import os

np.random.seed(42)
random.seed(42)

# 1) Simulate telemetry
T = 10000  # timesteps
start = datetime.datetime(2025, 10, 20, 0, 0)
timestamps = [start + datetime.timedelta(seconds=i * 1) for i in range(T)]

# base normal signals
latency = np.random.normal(loc=50, scale=5, size=T)  # ms
error_rate = np.random.beta(a=1.5, b=100, size=T) * 0.02  # fraction
active_conns = np.random.poisson(lam=200, size=T)
cpu = np.random.normal(loc=40, scale=5, size=T)  # percent
dns_failures = np.random.binomial(1, 0.0005, size=T)  # sporadic

df = pd.DataFrame(
    {
        "ts": timestamps,
        "latency_ms": latency,
        "error_rate": error_rate,
        "active_conns": active_conns,
        "cpu_percent": cpu,
        "dns_failures": dns_failures,
    }
)

# 2) Inject labeled anomalies at random times
anomaly_types = ["network_failure", "cpu_spike", "dns_issue", "monitoring_subsystem"]
labels = ["normal"] * T

injected_events = []
for i in range(60):  # inject 60 anomaly events
    t = np.random.randint(300, T - 300)
    kind = random.choice(anomaly_types)
    injected_events.append((t, kind))
    length = np.random.randint(5, 80)  # duration of event in seconds
    for j in range(t, min(T, t + length)):
        if kind == "network_failure":
            # high latency spikes, drop active connections, increased errors
            df.loc[j, "latency_ms"] += np.random.normal(300, 50)
            df.loc[j, "active_conns"] = np.maximum(
                0, df.loc[j, "active_conns"] - np.random.poisson(100)
            )
            df.loc[j, "error_rate"] += np.random.beta(2, 10) * 0.5
            df.loc[j, "dns_failures"] += np.random.binomial(1, 0.05)
        elif kind == "cpu_spike":
            df.loc[j, "cpu_percent"] += np.random.normal(40, 10)
            df.loc[j, "latency_ms"] += np.random.normal(80, 20)
            df.loc[j, "error_rate"] += np.random.beta(1, 5) * 0.1
        elif kind == "dns_issue":
            # increased dns failures that cause higher latencies and endpoint errors
            df.loc[j, "dns_failures"] += np.random.binomial(1, 0.6)
            df.loc[j, "latency_ms"] += np.random.normal(150, 40)
            df.loc[j, "error_rate"] += np.random.beta(1, 3) * 0.4
        elif kind == "monitoring_subsystem":
            # monitoring failure -> weird fluctuations in metrics, flapping values
            df.loc[j, "latency_ms"] += np.random.normal(120, 120)
            df.loc[j, "active_conns"] = np.maximum(
                0, df.loc[j, "active_conns"] + np.random.randint(-150, 150)
            )
            df.loc[j, "error_rate"] += np.random.beta(1, 8) * 0.3
        labels[j] = kind

df["label"] = labels

# Ensure all metrics are non-negative (fix for edge cases)
df["latency_ms"] = df["latency_ms"].clip(lower=0)
df["error_rate"] = df["error_rate"].clip(lower=0, upper=1.0)
df["active_conns"] = df["active_conns"].clip(lower=0)
df["cpu_percent"] = df["cpu_percent"].clip(lower=0, upper=100)

# 3) Feature engineering: add rolling statistics
window = 15
df["latency_roll_mean"] = df["latency_ms"].rolling(window=window, min_periods=1).mean()
df["latency_roll_std"] = (
    df["latency_ms"].rolling(window=window, min_periods=1).std().fillna(0)
)
df["error_roll_mean"] = df["error_rate"].rolling(window=window, min_periods=1).mean()
df["active_conns_roll_mean"] = (
    df["active_conns"].rolling(window=window, min_periods=1).mean()
)
df["cpu_roll_mean"] = df["cpu_percent"].rolling(window=window, min_periods=1).mean()
df["dns_failures_roll_sum"] = (
    df["dns_failures"].rolling(window=window, min_periods=1).sum()
)

feature_cols = [
    "latency_ms",
    "latency_roll_mean",
    "latency_roll_std",
    "error_rate",
    "error_roll_mean",
    "active_conns",
    "active_conns_roll_mean",
    "cpu_percent",
    "cpu_roll_mean",
    "dns_failures",
    "dns_failures_roll_sum",
]

# Ensure no NaN or inf values in features
for col in feature_cols:
    df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

X = df[feature_cols].values

# 4) Train IsolationForest on presumed-normal subset (first 60% without injected anomalies)
train_cut = int(0.6 * T)
normal_mask = (df.index < train_cut) & (df["label"] == "normal")
if normal_mask.sum() < 100:
    # fallback: use first 50%
    normal_mask = (df.index < int(0.5 * T)) & (df["label"] == "normal")

# Ensure we have enough training data
if normal_mask.sum() < 50:
    print(f"Warning: Only {normal_mask.sum()} normal samples for training. Using more data.")
    normal_mask = df["label"] == "normal"
    if normal_mask.sum() < 50:
        # Ultimate fallback: use first 1000 points regardless
        normal_mask = df.index < min(1000, len(df))

iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(X[normal_mask])

# anomaly scores (-1 anomaly, 1 normal)
iso_preds = iso.predict(X)
anomaly_flag = (iso_preds == -1).astype(int)
df["anomaly_flag"] = anomaly_flag

# 5) Evaluate detection on injected anomaly labels (binary)
true_anomaly = (df["label"] != "normal").astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_anomaly, df["anomaly_flag"], average="binary", zero_division=0
)
acc = accuracy_score(true_anomaly, df["anomaly_flag"])

# 6) Prepare dataset for root-cause classifier: only points labeled as anomalies
anom_df = df[df["label"] != "normal"].copy()

# Check if we have enough anomalies
if len(anom_df) == 0:
    print("Warning: No anomalies detected in dataset. Skipping classification training.")
    clf = None
else:
    # create balanced set by sampling some normal windows too
    normal_sample_size = min(len(anom_df), 2000, (df["label"] == "normal").sum())
    if normal_sample_size > 0:
        normal_sample = df[df["label"] == "normal"].sample(
            n=normal_sample_size, random_state=42
        )
        clf_df = pd.concat([anom_df, normal_sample])
    else:
        clf_df = anom_df.copy()

    # map labels to classes (including 'normal' as class 0)
    label_map = {
        "normal": 0,
        "network_failure": 1,
        "cpu_spike": 2,
        "dns_issue": 3,
        "monitoring_subsystem": 4,
    }
    clf_df["class"] = clf_df["label"].map(label_map)

    X_clf = clf_df[feature_cols].values
    y_clf = clf_df["class"].values

    # Check if we have multiple classes
    unique_classes = np.unique(y_clf)
    if len(unique_classes) < 2:
        print(f"Warning: Only {len(unique_classes)} class(es) available. Need at least 2 for classification.")
        clf = None
    else:
        # Use stratified split if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
            )
        except ValueError:
            # Fallback to non-stratified if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X_clf, y_clf, test_size=0.3, random_state=42
            )

        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        report = classification_report(
            y_test,
            y_pred,
            target_names=[k for k, v in sorted(label_map.items(), key=lambda x: x[1])],
            zero_division=0
        )
        conf = confusion_matrix(y_test, y_pred)

# 7) Define simulated mitigation policies
mitigation_policies = {
    "network_failure": lambda t: f"[{t}] ACTION: Reroute traffic away from affected AZ and provision extra load-balancers.",
    "cpu_spike": lambda t: f"[{t}] ACTION: Auto-scale compute pool (add instances) and throttle batch jobs.",
    "dns_issue": lambda t: f"[{t}] ACTION: Failover to secondary DNS providers, clear DNS caches.",
    "monitoring_subsystem": lambda t: f"[{t}] ACTION: Switch to backup monitoring plane, enable degraded-mode health checks.",
    "normal": lambda t: f"[{t}] ACTION: No action required.",
}

# 8) Simulate live detection + mitigation on a short window and show sample actions
simulation_window = df.iloc[4000:4200].copy()
actions = []

if clf is not None:
    for idx, row in simulation_window.iterrows():
        if row["anomaly_flag"] == 1:
            # predict root cause using classifier (use full feature vector)
            feat = row[feature_cols].values.reshape(1, -1)
            pred_class = clf.predict(feat)[0]
            # map back to label name
            inv_map = {v: k for k, v in label_map.items()}
            pred_label = inv_map.get(pred_class, "unknown")
            action = mitigation_policies.get(pred_label, mitigation_policies["normal"])(
                row["ts"]
            )
            actions.append((row["ts"], pred_label, action))

# 9) Plot a short subsection to visualize anomalies
plot_slice = df.iloc[3980:4220]
plt.figure(figsize=(12, 4))
plt.plot(plot_slice["ts"], plot_slice["latency_ms"], label="latency_ms", linewidth=1)
# mark anomalies detected by IsolationForest
anoms = plot_slice[plot_slice["anomaly_flag"] == 1]
if len(anoms) > 0:
    plt.scatter(anoms["ts"], anoms["latency_ms"], color='red', marker="x", s=40, label="Detected anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.title("Latency with detected anomalies (x marks)")
plt.legend()
plt.tight_layout()
plt.show()

# 10) Show sample of the simulated telemetry and produced actions
sample_display = df[
    [
        "ts",
        "latency_ms",
        "error_rate",
        "active_conns",
        "cpu_percent",
        "dns_failures",
        "label",
        "anomaly_flag",
    ]
].iloc[3990:4005].copy()

print("=== Detection performance on synthetic data ===")
print(
    f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {acc:.3f}"
)

if clf is not None:
    print("\n=== Root-cause classification report (test set) ===")
    print(report)
    print("Confusion matrix:\n", conf)
else:
    print("\n=== Root-cause classifier not trained (insufficient data) ===")

print("\n=== Sample telemetry rows ===")
sample_display.reset_index(drop=True, inplace=True)
display_df = sample_display.copy()

# use the display helper if available (Jupyter), otherwise print
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Sample telemetry (simulated)", display_df)
except Exception:
    print(display_df.to_string(index=False))

print("\n=== Simulated automated mitigation actions in window ===")
if len(actions) > 0:
    for t, label, action in actions[:20]:
        print(action)
else:
    print("No anomalies detected in simulation window.")

# Save small artifacts for download (if user wants)
# Create directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "simulated_telemetry_sample.csv")

try:
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full simulated telemetry to {csv_path} (you can download it).")
except Exception as e:
    print(f"\nCould not save CSV file: {e}")