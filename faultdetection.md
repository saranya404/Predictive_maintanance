# Fault Detection Notebook Explanation

AIzaSyCQpZXBndh9hPg477rTCuGMK6NsSftpAek
This document explains the logic of each code cell in the notebook and the reasoning behind the feature engineering, windowing, and modeling choices.

## Cell 1 - Imports

This cell imports the libraries used in the notebook:
- `numpy`, `scipy.io`, `glob`, `os`: data handling, file discovery, and loading MATLAB `.mat` files.
- `matplotlib.pyplot`: plotting (not used later, but available).
- `kurtosis`, `skew`: higher-order statistics for time-domain signal characterization.
- `StandardScaler`: feature scaling (zero mean, unit variance).
- `OneClassSVM`: anomaly detection model trained on normal data only.

## Cell 2 - `extract_features(...)`

This function converts a raw vibration signal into a set of numeric features. It uses overlapping time windows and returns one feature vector per window.

### Why windowing is used
- Vibration signals are non-stationary; their statistics change over time.
- Windowing splits a long signal into shorter segments so each segment is closer to stationary.
- Overlap (50%) increases the number of training samples without throwing away data between windows.
- Window size (1024 samples at 12 kHz) provides a balance between time resolution and frequency resolution.

### Time-domain features
Each window computes:
- **Mean**: average value. A healthy vibration signal should be centered near zero; drift can indicate imbalance or sensor bias.
- **Standard deviation (STD)**: spread of amplitudes. Higher spread often means higher vibration energy.
- **RMS**: root-mean-square energy. Useful for overall vibration severity.
- **Peak value**: maximum absolute amplitude. Captures transient spikes and impacts.
- **Kurtosis**: measures tail heaviness. High kurtosis suggests impulsive events (bearing faults).
- **Skewness**: measures asymmetry. Can indicate non-symmetric impacts or bias.
- **Crest factor**: peak / RMS. High crest factor implies sharp impacts relative to average energy.

### Frequency-domain features
- **FFT magnitude** gives the spectral energy distribution.
- **Peak frequency**: dominant frequency of vibration. Faults can shift dominant frequencies.
- **Spectral centroid**: energy-weighted average frequency, a measure of where the spectrum is "centered".

### Band energy features
- **Low band (< 1000 Hz)**: typically fundamental and low-order harmonics.
- **Mid band (1000-3000 Hz)**: common range for some bearing defect signatures.
- **High band (>= 3000 Hz)**: impulsive content and higher harmonics.

These band energies help capture changes in spectral distribution caused by faults.

## Cell 3 - `load_fault_detection_data(...)`

This function reads all `.mat` files, extracts the drive-end vibration channel, and builds training and testing datasets.

Logic:
- Find all `.mat` files under the dataset folder (recursive).
- For each file:
  - Load the MATLAB file and find a key containing `DE_time` (drive-end signal).
  - If the file path contains `Normal`, split the signal into 80% train and 20% test.
  - Fault files are used entirely as anomaly test data.
- Feature extraction is applied to each segment.
- Outputs:
  - `X_train`: features from normal training windows.
  - `X_test_normal`: features from normal test windows.
  - `X_test_anomaly`: features from faulty windows.

## Cell 4 - Load and summarize data

This cell:
- Sets the dataset path.
- Calls `load_fault_detection_data(...)`.
- Prints the shapes of the resulting feature matrices.

This verifies that the data pipeline is working and that enough samples were created by windowing.

## Cell 5 - Feature scaling

`StandardScaler` is applied to all features:
- Each feature is centered to zero mean and scaled to unit variance.
- SVMs are sensitive to feature scales, and this step prevents features with large numeric ranges from dominating the model.
- Scaling is fit on training data only, then applied to test sets.

## Cell 6 - One-Class SVM training

This cell trains the anomaly detector:
- `OneClassSVM` learns a boundary around normal data.
- `kernel="rbf"`: captures non-linear boundaries.
- `nu=0.01`: expects about 1% outliers in training; keeps boundary tight.
- `gamma="scale"`: standard heuristic based on feature variance.

The model is trained only on normal data, so it can flag deviations as faults.

## Cell 7 - Evaluation

Predictions:
- `1` means normal, `-1` means anomaly.

Metrics:
- **Fault Detection Rate (Recall)**: $TP / (TP + FN)$, how many faults were detected.
- **False Alarm Rate**: $FP / (FP + TN)$, how often normal data is flagged as faulty.

This provides a quick performance snapshot for anomaly detection.

## Cell 8 - Confusion matrix printout

This cell prints counts of:
- Fault detected (TP)
- Fault missed (FN)
- False alarm (FP)
- Normal correct (TN)

It is a readable summary of model behavior.

## Cell 9 - Empty cell

This cell is empty. It does not affect execution.

## Cell 10 - Score ranges (missing variables)

This cell attempts to print:
- `normal_scores`
- `fault_scores`

These variables are not defined in the notebook, so running this cell will raise a `NameError`. If you want this output, the notebook needs to compute decision scores, for example using `model.decision_function(...)`.

## Cell 11 - Save model and scaler

This cell saves the trained artifacts:
- `anomaly_model.pkl`: the One-Class SVM.
- `anomaly_scaler.pkl`: the fitted `StandardScaler`.

These files allow you to load the trained detector later and apply the same scaling to new data.
