import streamlit as st
import numpy as np
import scipy.io
import joblib
import os
import glob
import time
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# =========================================
# 1. SETUP & HELPER FUNCTIONS
# =========================================
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Load Models (Cache them for speed)
@st.cache_resource
def load_models():
    try:
        anomaly_model = joblib.load('anomaly_model.pkl')
        anomaly_scaler = joblib.load('anomaly_scaler.pkl')
        classifier_model = joblib.load('classifier_model.pkl')
        classifier_scaler = joblib.load('classifier_scaler.pkl')
        return anomaly_model, anomaly_scaler, classifier_model, classifier_scaler
    except:
        return None, None, None, None

# Feature Extraction (Same as your v3 code)
def extract_features(signal, fs=12000):
    # Time Domain
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    peak_val = np.max(np.abs(signal))
    kurt_val = kurtosis(signal)
    skew_val = skew(signal)
    crest_factor = peak_val / (rms_val + 1e-6)

    # Frequency Domain
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    spec_centroid = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)

    # Energy Bands
    band_low = np.sum(fft_vals[fft_freqs < 1000]**2)
    band_mid = np.sum(fft_vals[(fft_freqs >= 1000) & (fft_freqs < 3000)]**2)
    band_high = np.sum(fft_vals[fft_freqs >= 3000]**2)

    return np.array([mean_val, std_val, rms_val, kurt_val, skew_val, crest_factor,
                     peak_freq, spec_centroid, band_low, band_mid, band_high]).reshape(1, -1)

# Noise Injection Function
def add_noise(signal, snr_db):
    if snr_db >= 50: return signal # 50dB is practically clean
    power_sig = np.mean(signal ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(power_noise), signal.shape)
    return signal + noise

# =========================================
# 2. SIDEBAR CONTROLS
# =========================================
st.sidebar.title("⚙️ Simulation Control")
st.sidebar.markdown("Control the environment conditions.")

# Data Source Selection
data_mode = st.sidebar.radio("Data Source", ["Simulate Live Sensor", "Upload File"])

# Noise Control (The Stress Test)
st.sidebar.subheader("Environmental Noise")
noise_level = st.sidebar.slider("Signal-to-Noise Ratio (dB)", min_value=-5, max_value=50, value=50, step=5)
st.sidebar.caption("Left (-5dB) = Extreme Noise | Right (50dB) = Clean Signal")

# =========================================
# 3. MAIN DASHBOARD UI
# =========================================
st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("Real-time Health Monitoring System for Industrial Motors")

# Load Models
anomaly_clf, anomaly_scl, fault_clf, fault_scl = load_models()

if anomaly_clf is None:
    st.error("⚠️ Models not found! Please run 'joblib.dump' in your notebooks to save 'anomaly_model.pkl', etc.")
    st.stop()

# Placeholders for Layout
col1, col2 = st.columns([2, 1])
status_placeholder = col2.empty()
metrics_placeholder = col1.empty()
plot_placeholder = st.empty()

# Logic to get the signal
signal = None

if data_mode == "Simulate Live Sensor":
    scenario = st.sidebar.selectbox("Simulate Condition", ["Normal Operation", "Inner Race Fault", "Outer Race Fault", "Ball Fault"])
    
    # HEURISTIC: Load a random file corresponding to the scenario from your dataset
    # NOTE: You must update 'dataset_path' to your actual CWRU folder location
    dataset_path = "./CWRU-dataset" 
    
    if scenario == "Normal Operation": pattern = "Normal"
    elif scenario == "Inner Race Fault": pattern = "IR"
    elif scenario == "Outer Race Fault": pattern = "OR"
    elif scenario == "Ball Fault": pattern = "B"
    
    try:
        files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
        relevant_files = [f for f in files if pattern in f]
        if relevant_files:
            # Pick a random file and a random window within it
            chosen_file = np.random.choice(relevant_files)
            mat = scipy.io.loadmat(chosen_file)
            key = next((k for k in mat.keys() if "DE_time" in k), None)
            full_signal = mat[key].flatten()
            
            # Simulate streaming by taking a random slice
            start_idx = np.random.randint(0, len(full_signal) - 1024)
            signal = full_signal[start_idx : start_idx + 1024]
        else:
            st.error(f"No files found for pattern '{pattern}' in {dataset_path}")
    except Exception as e:
        st.error(f"Error loading simulation data: {e}")

elif data_mode == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload .mat file", type=["mat"])
    if uploaded_file:
        try:
            mat = scipy.io.loadmat(uploaded_file)
            key = next((k for k in mat.keys() if "DE_time" in k), None)
            signal = mat[key].flatten()[:1024] # Just take first window
        except:
            st.error("Invalid .mat file structure")

# =========================================
# 4. PROCESSING & VISUALIZATION
# =========================================
if signal is not None:
    # 1. Apply Noise (Simulation Step)
    noisy_signal = add_noise(signal, noise_level)
    
    # 2. Extract Features
    features_raw = extract_features(noisy_signal)
    
    # 3. Anomaly Detection
    feats_anomaly_scaled = anomaly_scl.transform(features_raw)
    is_anomaly = anomaly_clf.predict(feats_anomaly_scaled)[0] == -1 # -1 is Anomaly in OneClassSVM
    
    # 4. Fault Classification (Only if Anomaly)
    diagnosis = "Normal"
    confidence = 1.0
    
    if is_anomaly:
        feats_class_scaled = fault_scl.transform(features_raw)
        pred_idx = fault_clf.predict(feats_class_scaled)[0]
        probs = fault_clf.predict_proba(feats_class_scaled)[0]
        
        labels = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
        diagnosis = labels[pred_idx]
        confidence = probs[pred_idx]

    # --- UI UPDATES ---
    
    # A. Signal Plot
    with plot_placeholder.container():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Time Domain
        ax1.plot(noisy_signal, color='blue', alpha=0.7)
        ax1.set_title("Vibration Signal (Time Domain)")
        ax1.set_ylim([-2, 2]) # Fix y-axis for stability visually
        
        # Frequency Domain (FFT)
        fft_vals = np.abs(np.fft.rfft(noisy_signal))
        freqs = np.fft.rfftfreq(len(noisy_signal), 1/12000)
        ax2.plot(freqs, fft_vals, color='orange')
        ax2.set_title("Frequency Spectrum (FFT)")
        ax2.set_xlabel("Frequency (Hz)")
        
        st.pyplot(fig)

    # B. Status Indicators
    with status_placeholder.container():
        if not is_anomaly:
            st.success("## ✅ SYSTEM STATUS: HEALTHY")
            st.metric("Anomaly Score", "Normal")
        else:
            st.error(f"## 🚨 WARNING: FAULT DETECTED")
            st.warning(f"### Diagnosis: **{diagnosis}**")
            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
            
    # C. Real-Time Metrics
    with metrics_placeholder.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMS (Vibration)", f"{features_raw[0][2]:.4f}")
        m2.metric("Peak Freq", f"{features_raw[0][6]:.1f} Hz")
        m3.metric("Crest Factor", f"{features_raw[0][5]:.2f}")
        m4.metric("Noise Level", f"{noise_level} dB")
        
    # D. Explainability (Simple Feature Highlight)
    st.divider()
    st.subheader("🔍 Diagnostics Analysis")
    st.write("Feature values extracted from the current signal window:")
    st.dataframe(features_raw, hide_index=True)
    
else:
    st.info("👈 Select a simulation scenario or upload a file to begin.")