import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

print("Loading Data...")
df_sensors = pd.read_csv('secom.data', sep="\s+", header=None)
df_labels = pd.read_csv('secom_labels.data', sep="\s+", header=None)
df_final = pd.concat([df_sensors, df_labels[0]], axis=1)
df_final.columns = [f"Sensor_{i+1}" for i in range(590)] + ['Pass_Fail']

df_cleaned = df_final.dropna(axis=1, thresh=len(df_final)*0.6)
df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() > 1]
df_cleaned = df_cleaned.fillna(0)

X = df_cleaned.drop(columns=['Pass_Fail'])
y = df_cleaned['Pass_Fail']
y = y.apply(lambda x: 0 if x == -1 else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

print("\nTesting with 'Strict Inspector' Threshold...")
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.10
y_pred = (probs > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("-" * 30)
print(f"FAILURES CAUGHT: {cm[1][1]} out of {cm[1][0] + cm[1][1]}")
print("-" * 30)

print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
X_test_sample = X_test.iloc[:50]
shap_values = explainer.shap_values(X_test_sample)

if isinstance(shap_values, list):
    class_1_values = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    class_1_values = shap_values[:, :, 1]
else:
    class_1_values = shap_values

probs_sample = model.predict_proba(X_test_sample)[:, 1]
worst_wafer_idx = np.argmax(probs_sample)

print(f"\nAnalyzing Wafer #{worst_wafer_idx} from sample...")
print(f"Model Confidence: {probs_sample[worst_wafer_idx] * 100:.2f}% Fail")

wafer_shap_vals = class_1_values[worst_wafer_idx]
wafer_shap_vals = np.array(wafer_shap_vals).flatten()

top_3_indices = np.argsort(np.abs(wafer_shap_vals))[-3:][::-1]

print("\nTOP 3 SENSORS CAUSING THIS FAILURE:")
print("-" * 30)
for i in top_3_indices:
    if i < len(X_test.columns):
        sensor_name = X_test.columns[i]
        impact_score = wafer_shap_vals[i]
        print(f" -> {sensor_name} (Impact: {impact_score:.4f})")
print("-" * 30)


print("\n" + "="*40)
print("       AI INSPECTOR REPORT GENERATOR")
print("="*40)

top_sensors = [X_test.columns[i] for i in top_3_indices]

prompt = f"""
You are a Senior Process Engineer at a Semiconductor Fab (TSMC/Intel).
My Virtual Metrology AI has detected a potential failure on a wafer.

The model identified these 3 sensors as the most critical factors:
1. {top_sensors[0]}
2. {top_sensors[1]}
3. {top_sensors[2]}

TASK:
Write a "Root Cause Analysis" report for a junior technician.
Since the sensor names are anonymized, please INVENT realistic technical identities for them (e.g., "Etch Chamber Pressure", "Deposition Temp", "Helium Flow").
Explain how an anomaly in these specific areas would cause a chip failure.
End with a "Recommended Maintenance Action".
"""

print("COPY AND PASTE THE TEXT BELOW INTO CHATGPT OR GEMINI:")
print("-" * 40)
print(prompt)
print("-" * 40)