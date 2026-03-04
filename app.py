from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from fpdf import FPDF
import datetime
import io
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
X_data = None

def train_model():
    global model, X_data
    print("Training Model... Please wait...")

    secom_data_path = os.path.join(BASE_DIR, 'secom.data')
    secom_labels_path = os.path.join(BASE_DIR, 'secom_labels.data')

    if not os.path.exists(secom_data_path) or not os.path.exists(secom_labels_path):
        print(f"ERROR: Data files not found in {BASE_DIR}")
        print("Make sure secom.data and secom_labels.data are in the same folder as app.py")
        return

    try:
        df_sensors = pd.read_csv(secom_data_path, sep=r"\s+", header=None)
        df_labels = pd.read_csv(secom_labels_path, sep=r"\s+", header=None)

        df_final = pd.concat([df_sensors, df_labels[0]], axis=1)
        df_final.columns = [f"Sensor_{i+1}" for i in range(590)] + ['Pass_Fail']

        df_cleaned = df_final.dropna(axis=1, thresh=len(df_final) * 0.6)
        df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() > 1]
        df_cleaned = df_cleaned.fillna(0)

        X = df_cleaned.drop(columns=['Pass_Fail'])
        y = df_cleaned['Pass_Fail']
        y = y.apply(lambda x: 0 if x == -1 else 1)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)
        X_data = X
        print("Model Trained and Ready!")

    except Exception as e:
        print(f"Training failed: {e}")
        model = None
        X_data = None

train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    if model is None:
        return jsonify({'ready': False, 'message': 'Model not trained. Check secom.data files are present.'})
    return jsonify({'ready': True, 'message': 'Model ready.'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not trained. Ensure secom.data and secom_labels.data exist in the project folder.'}), 503

    if X_data is None:
        return jsonify({'error': 'Training data reference is missing.'}), 503

    try:
        data = request.json

        if not data:
            return jsonify({'error': 'No JSON data received.'}), 400

        for key in ['s_487', 's_96', 's_481']:
            if key not in data:
                return jsonify({'error': f'Missing sensor value: {key}'}), 400

        input_wafer = X_data.mean().to_frame().T

        if 'Sensor_487' in input_wafer.columns:
            input_wafer['Sensor_487'] = float(data['s_487'])
        if 'Sensor_96' in input_wafer.columns:
            input_wafer['Sensor_96'] = float(data['s_96'])
        if 'Sensor_481' in input_wafer.columns:
            input_wafer['Sensor_481'] = float(data['s_481'])

        prob_fail = model.predict_proba(input_wafer)[0][1]

        return jsonify({'probability': float(prob_fail)})

    except ValueError as e:
        return jsonify({'error': f'Invalid sensor value: {str(e)}'}), 400
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    if model is None:
        return jsonify({'error': 'Model is not trained.'}), 503

    try:
        data = request.json

        if not data:
            return jsonify({'error': 'No JSON data received.'}), 400

        prob = float(data['prob'])
        sensors = data['sensors']

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="SEMICONDUCTOR VIRTUAL METROLOGY REPORT", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=10)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        pdf.cell(200, 10, txt=f"Date Generated: {timestamp}", ln=True)
        pdf.cell(200, 10, txt="Batch ID: SIM-2026-X99", ln=True)
        pdf.line(10, 45, 200, 45)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="1. DIAGNOSTIC SUMMARY", ln=True)
        pdf.set_font("Arial", size=11)

        status = "CRITICAL FAIL" if prob > 0.1 else "PASS"

        pdf.cell(200, 8, txt=f"Prediction Status: {status}", ln=True)
        pdf.cell(200, 8, txt=f"Failure Probability: {prob * 100:.2f}%", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="2. SENSOR TELEMETRY", ln=True)
        pdf.set_font("Courier", size=10)

        pdf.cell(200, 6, txt=f"[*] RF Generator Power (S_487):       {sensors['s_487']} sigma", ln=True)
        pdf.cell(200, 6, txt=f"[*] Helium Cooling Flow (S_96):       {sensors['s_96']} sigma", ln=True)
        pdf.cell(200, 6, txt=f"[*] Electrostatic Chuck Volt (S_481): {sensors['s_481']} sigma", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="3. AI ROOT CAUSE ANALYSIS", ln=True)
        pdf.set_font("Arial", size=11)

        vals = {k: abs(float(v)) for k, v in sensors.items()}
        worst_key = max(vals, key=vals.get)
        sensor_map = {
            's_487': "RF Generator Power",
            's_96': "Helium Cooling Flow",
            's_481': "Chuck Voltage"
        }

        pdf.multi_cell(0, 8, txt=(
            f"Primary Anomaly Detected: {sensor_map[worst_key]}.\n\n"
            f"Technical Assessment: The model identified significant statistical drift in {sensor_map[worst_key]}. "
            f"In a plasma etch process, this anomaly typically leads to incomplete trench depth or sidewall profile bowing. "
            f"Yield loss is highly probable if the tool continues operation."
        ))
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="4. RECOMMENDED ACTIONS", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 8, txt="[ ] IMMEDIATE: Halt tool execution.", ln=True)
        pdf.cell(200, 8, txt=f"[ ] INSPECT: Hardware diagnostics on {sensor_map[worst_key]}.", ln=True)
        pdf.cell(200, 8, txt="[ ] VERIFY: Run calibration wafer SOP-884.", ln=True)

        buffer = io.BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        buffer.write(pdf_bytes)
        buffer.seek(0)

        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"Report_{safe_timestamp}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)