import streamlit as st
import numpy as np
import scipy.io
import joblib
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import google.generativeai as genai
import serial
import serial.tools.list_ports
import time
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# =========================================
# 1. PAGE SETUP & SESSION STATE
# =========================================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

# Initialize session state for live hardware
if 'serial_conn' not in st.session_state:
    st.session_state.serial_conn = None
if 'live_model' not in st.session_state:
    st.session_state.live_model = None
if 'live_scaler' not in st.session_state:
    st.session_state.live_scaler = None
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = None

# =========================================
# 2. GEMINI CONFIG & ERROR HANDLING
# =========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"Model Initialization Error: {e}")

# =========================================
# 3. LOAD STATIC CWRU MODELS (CACHED)
# =========================================
@st.cache_resource
def load_models():
    anomaly_model = joblib.load("anomaly_model.pkl")
    anomaly_scaler = joblib.load("anomaly_scaler.pkl")
    classifier_model = joblib.load("classifier_model.pkl")
    classifier_scaler = joblib.load("classifier_scaler.pkl")
    return anomaly_model, anomaly_scaler, classifier_model, classifier_scaler

# =========================================
# 4. FEATURE EXTRACTION (DUAL MODE)
# =========================================
def extract_features(signal, fs=12000):
    """Original CWRU Feature Extractor (12,000 Hz)"""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    peak_val = np.max(np.abs(signal))
    kurt_val = kurtosis(signal) if np.std(signal) > 0 else 0
    skew_val = skew(signal) if np.std(signal) > 0 else 0
    crest_factor = peak_val / (rms_val + 1e-6)

    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)

    peak_freq = fft_freqs[np.argmax(fft_vals)]
    spec_centroid = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)

    band_low = np.sum(fft_vals[fft_freqs < 1000]**2)
    band_mid = np.sum(fft_vals[(fft_freqs >= 1000) & (fft_freqs < 3000)]**2)
    band_high = np.sum(fft_vals[fft_freqs >= 3000]**2)

    return np.array([
        mean_val, std_val, rms_val, kurt_val, skew_val, crest_factor,
        peak_freq, spec_centroid, band_low, band_mid, band_high
    ]).reshape(1, -1)

def extract_live_features(signal, fs=200):
    """New Arduino Feature Extractor (200 Hz)"""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    peak_val = np.max(np.abs(signal))
    kurt_val = kurtosis(signal) if np.std(signal) > 0 else 0
    skew_val = skew(signal) if np.std(signal) > 0 else 0
    crest_factor = peak_val / (rms_val + 1e-6)

    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)

    peak_freq = fft_freqs[np.argmax(fft_vals)]
    spec_centroid = np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-6)

    # Scaled down frequency bands for a 100Hz Nyquist limit
    band_low = np.sum(fft_vals[fft_freqs < 20]**2)
    band_mid = np.sum(fft_vals[(fft_freqs >= 20) & (fft_freqs < 60)]**2)
    band_high = np.sum(fft_vals[fft_freqs >= 60]**2)

    return np.array([
        mean_val, std_val, rms_val, kurt_val, skew_val, crest_factor,
        peak_freq, spec_centroid, band_low, band_mid, band_high
    ]).reshape(1, -1)

def add_noise(signal, snr_db):
    if snr_db >= 50:
        return signal
    power_sig = np.mean(signal ** 2)
    power_noise = power_sig / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(power_noise), signal.shape)
    return signal + noise

def explain_decision(diagnosis, anomaly_score, features):
    rms, kurt, crest, peak_f = features[0][2], features[0][3], features[0][5], features[0][6]
    prompt = f"""
    You are an Industrial Reliability Engineer. Provide a technical analysis of this bearing condition.
    DATA SUMMARY:
    - Diagnosis: {diagnosis}
    - Anomaly Score: {anomaly_score:.3f}
    - RMS (Total Energy): {rms:.4f}
    - Kurtosis (Impact Spikes): {kurt:.2f}
    - Crest Factor: {crest:.2f}
    - Peak Frequency: {peak_f:.1f} Hz
    Please provide:
    1. A brief explanation of why the metrics led to this diagnosis.
    2. A specific MAINTENANCE RECOMMENDATION.
    3. The likely root cause based on the {diagnosis}.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis failed. Check API quota. Error: {e}"

# =========================================
# 5. UI LAYOUT
# =========================================
st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("Real-time health monitoring for rotating machinery using AI & Vibration Analysis")

tab_live, tab_cwru = st.tabs(["⚙️ Live Hardware Demo", "📊 CWRU Industrial Benchmark"])

# =========================================
# TAB 1: LIVE HARDWARE DEMO
# =========================================
with tab_live:
    st.header("Real-Time Hardware Inference")
    st.write("Connect your rig, record a healthy baseline, and monitor for anomalies live.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Hardware Connection")
        ports = [port.device for port in serial.tools.list_ports.comports()]
        selected_port = st.selectbox("Select Arduino COM Port", ports if ports else ["No devices found"])

        if st.button("Connect to Arduino"):
            if selected_port != "No devices found":
                try:
                    if st.session_state.serial_conn:
                        st.session_state.serial_conn.close()
                    st.session_state.serial_conn = serial.Serial(selected_port, 115200, timeout=1)
                    time.sleep(2) 
                    st.success(f"Connected to {selected_port}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
            else:
                st.warning("Please plug in the Arduino.")

        st.divider()
        st.subheader("2. Model Training")
        st.write("Turn on the motor. Let it spin up to full speed, then click Record.")

        if st.button("🔴 Record Baseline (10s)"):
            if st.session_state.serial_conn is not None:
                st.info("Recording healthy vibration data... Do not touch the motor!")

                progress_bar = st.progress(0, text="Recording Baseline: 0%")
                graph_placeholder = st.empty()
                raw_data = []

                st.session_state.serial_conn.reset_input_buffer()
                
                start_time = time.time()
                duration = 15.0
                last_update_time = start_time

                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        break 
                        
                    progress_pct = min(int((elapsed / duration) * 100), 100)
                    progress_bar.progress(progress_pct, text=f"Recording Baseline: {progress_pct}%")

                    try:
                        # Robust serial reading
                        st.session_state.serial_conn.timeout = 0.5 
                        raw_bytes = st.session_state.serial_conn.readline()
                        line = raw_bytes.decode('utf-8', errors='ignore').strip()
                        
                        if line and "X" not in line: 
                            parts = line.split(',')
                            if len(parts) == 3: # Strict check for exactly 3 numbers
                                x, y, z = map(float, parts)
                                magnitude = np.sqrt(x**2 + y**2 + z**2) - 9.8
                                raw_data.append(magnitude)

                                if time.time() - last_update_time > 0.5:
                                    graph_placeholder.line_chart(raw_data[-200:])
                                    last_update_time = time.time()
                    except Exception:
                        pass # Ignore any mangled lines completely

                progress_bar.progress(100, text="Recording Complete! Extracting features...")
                st.session_state.baseline_data = np.array(raw_data)

                # Safety Check
                if len(st.session_state.baseline_data) < 256:
                    st.error(f"🚨 Recording Failed: Only collected {len(st.session_state.baseline_data)} samples.")
                    st.warning("We need at least 256 samples. Ensure the Arduino IDE Serial Monitor is closed and the Arduino 'delay()' is set to 5ms.")
                else:
                    st.write(f"Collected {len(st.session_state.baseline_data)} samples. Extracting features and training One-Class SVM...")
                    features_list = []
                    window_size = 256
                    step = 64

                    for i in range(0, len(st.session_state.baseline_data) - window_size, step):
                        window = st.session_state.baseline_data[i : i + window_size]
                        feats = extract_live_features(window)
                        features_list.append(feats[0])

                    X_train = np.array(features_list)

                    live_scaler = StandardScaler()
                    X_train_scaled = live_scaler.fit_transform(X_train)

                    live_svm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
                    live_svm.fit(X_train_scaled)

                    st.session_state.live_scaler = live_scaler
                    st.session_state.live_model = live_svm

                    st.success("✅ Live Baseline Model Trained Successfully!")
            else:
                st.error("Please connect to the Arduino first.")

    with col2:
        st.subheader("3. Live Monitoring Dashboard")
        if st.session_state.live_model is not None:
            if st.button("▶️ Start Live Monitoring"):
                st.write("Monitoring active... Try tapping the motor to simulate a fault!")

                status_placeholder = st.empty()
                chart_placeholder = st.empty()

                while True:
                    live_window = []
                    st.session_state.serial_conn.reset_input_buffer()
                    
                    while len(live_window) < 256:
                        try:
                            # Use the same robust reading logic here
                            st.session_state.serial_conn.timeout = 0.5
                            raw_bytes = st.session_state.serial_conn.readline()
                            line = raw_bytes.decode('utf-8', errors='ignore').strip()
                            
                            if line and "X" not in line:
                                parts = line.split(',')
                                if len(parts) == 3:
                                    x, y, z = map(float, parts)
                                    magnitude = np.sqrt(x**2 + y**2 + z**2) - 9.8
                                    live_window.append(magnitude)
                        except Exception:
                            pass

                    live_signal = np.array(live_window)
                    live_feats = extract_live_features(live_signal)
                    scaled_live_feats = st.session_state.live_scaler.transform(live_feats)

                    score = st.session_state.live_model.decision_function(scaled_live_feats)[0]
                    threshold = st.slider(
                        "🚨 Alarm Sensitivity (Lower = Less Sensitive)",
                        min_value=-10.0,
                        max_value=0.0,
                        value=-1.0,
                        step=0.5,
                        key="sensitivity_slider"
                    )
                    
                    is_anomaly = score < threshold 

                    if is_anomaly:
                        status_placeholder.error(f"🚨 ANOMALY DETECTED! | Score: {score:.3f}")
                    else:
                        status_placeholder.success(f"✅ SYSTEM HEALTHY | Score: {score:.3f}")

                    fft_vals = np.abs(np.fft.rfft(live_signal))
                    freqs = np.fft.rfftfreq(len(live_signal), 1/200)

                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(freqs, fft_vals, color='red' if is_anomaly else 'green')
                    ax.set_title("Live Frequency Spectrum")
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylim(0, max(fft_vals) + 10) 
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
        else:
            st.info("Record a baseline to unlock the monitoring dashboard.")

# =========================================
# TAB 2: CWRU BENCHMARK (ORIGINAL CODE)
# =========================================
with tab_cwru:
    st.sidebar.title("⚙️ CWRU Simulation Controls")
    data_mode = st.sidebar.radio("Data Source", ["Simulate Live Sensor", "Upload File"])
    noise_level = st.sidebar.slider("Signal-to-Noise Ratio (dB)", -5, 50, 50, 5)

    try:
        anomaly_clf, anomaly_scl, fault_clf, fault_scl = load_models()
    except Exception as e:
        st.error(f"Error loading CWRU pickle files: {e}")
        st.stop()

    signal = None

    if data_mode == "Simulate Live Sensor":
        scenario = st.sidebar.selectbox(
            "Simulated Condition",
            ["Normal Operation", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
        )

        dataset_path = "./CWRU-dataset"
        pattern_map = {"Normal Operation": "Normal", "Ball Fault": "B", "Inner Race Fault": "IR", "Outer Race Fault": "OR"}
        pattern = pattern_map[scenario]

        files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
        relevant_files = [f for f in files if pattern in f]

        if relevant_files:
            chosen_file = np.random.choice(relevant_files)
            mat = scipy.io.loadmat(chosen_file)
            key = next(k for k in mat if "DE_time" in k)
            full_signal = mat[key].flatten()
            idx = np.random.randint(0, len(full_signal) - 1024)
            signal = full_signal[idx:idx+1024]
        else:
            st.warning(f"No dataset files found with pattern '{pattern}'")

    elif data_mode == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload .mat file", type=["mat"])
        if uploaded_file:
            mat = scipy.io.loadmat(uploaded_file)
            key = next(k for k in mat if "DE_time" in k)
            signal = mat[key].flatten()[:1024]

    if signal is not None:
        noisy_signal = add_noise(signal, noise_level)
        features = extract_features(noisy_signal)

        scaled_features = anomaly_scl.transform(features)
        anomaly_score = anomaly_clf.decision_function(scaled_features)[0]

        ANOMALY_THRESHOLD = -0.15
        is_anomaly = anomaly_score < ANOMALY_THRESHOLD
        diagnosis = "Normal"
        confidence = 1.0

        if is_anomaly:
            class_scaled = fault_scl.transform(features)
            pred_idx = fault_clf.predict(class_scaled)[0]
            probs = fault_clf.predict_proba(class_scaled)[0]

            labels = ["Normal", "Ball Fault", "Inner Race Fault", "Outer Race Fault"]
            predicted_label = labels[pred_idx]

            if predicted_label == "Normal":
                diagnosis = "Unidentified Anomaly"
                confidence = 0.5
            else:
                diagnosis = predicted_label
                confidence = probs[pred_idx]

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            fig1, ax1 = plt.subplots()
            ax1.plot(noisy_signal, color='#1f77b4')
            ax1.set_title("Time Domain Signal")
            ax1.set_xlabel("Samples")
            ax1.set_ylabel("Amplitude")
            st.pyplot(fig1)

        with col_v2:
            fig2, ax2 = plt.subplots()
            fft_vals = np.abs(np.fft.rfft(noisy_signal))
            freqs = np.fft.rfftfreq(len(noisy_signal), 1/12000)
            ax2.plot(freqs, fft_vals, color='#ff7f0e')
            ax2.set_title("Frequency Spectrum")
            ax2.set_xlabel("Frequency (Hz)")
            st.pyplot(fig2)

        if not is_anomaly:
            st.success("✅ SYSTEM STATUS: HEALTHY")
            st.metric("Anomaly Score", f"{anomaly_score:.3f}")
        else:
            st.error(f"🚨 FAULT DETECTED: {diagnosis.upper()}")
            st.metric("Anomaly Score", f"{anomaly_score:.3f}", delta_color="inverse")
            st.progress(confidence, text=f"Classification Confidence: {confidence:.2%}")

        st.subheader("Key Vibration Indicators")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMS (Energy)", f"{features[0][2]:.4f}")
        c2.metric("Peak Frequency", f"{features[0][6]:.1f} Hz")
        c3.metric("Crest Factor", f"{features[0][5]:.2f}")
        c4.metric("Kurtosis", f"{features[0][3]:.2f}")

        st.divider()
        st.subheader("🧠 AI Technical Analysis & Recommendation")

        if st.button("Generate Expert Report"):
            with st.spinner("Gemini is analyzing vibration signatures..."):
                explanation = explain_decision(diagnosis, anomaly_score, features)
                st.info(explanation)
        else:
            st.write("Click the button above to generate maintenance actions.")
    else:
        st.info("👈 Select a simulation scenario or upload a file to begin.")