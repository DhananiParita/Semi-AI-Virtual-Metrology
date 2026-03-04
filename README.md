<<<<<<< HEAD
# Semi-AI Virtual Metrology

An AI-powered semiconductor defect detection system that predicts wafer yield loss using a Random Forest model trained on real fab sensor data (SECOM dataset). Instead of waiting 4–6 hours for physical metrology, this tool gives an instant prediction from 3 key sensor inputs.

---

## What It Does

- Takes 3 sensor readings as input (RF Power, Helium Flow, Chuck Voltage)
- Runs them through a trained Random Forest classifier
- Predicts the probability of wafer failure in real time
- Displays results on a live radar chart with a log terminal
- Exports a full diagnostic PDF report with root cause analysis

---

## Project Structure

```
├── app.py                  # Flask backend — model training + API routes
├── main.py                 # Standalone analysis script with SHAP explainability
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── favicon.png
├── secom.data              # ⚠️ NOT included — download separately (see below)
├── secom_labels.data       # ⚠️ NOT included — download separately (see below)
├── requirements.txt
└── README.md
```

---

## Getting the Data Files

The SECOM dataset is **not included** in this repository due to file size. You must download it manually.

1. Go to the UCI Machine Learning Repository:
   **https://archive.ics.uci.edu/dataset/179/secom**

2. Download the dataset and extract it

3. Place these two files in the **root of the project folder** (same level as `app.py`):
   - `secom.data`
   - `secom_labels.data`

Without these files the model will not train and the app will not work.

---

## Requirements

- Python 3.8 or higher
- pip

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data files

Download `secom.data` and `secom_labels.data` from the UCI link above and place them in the project root folder.

Your folder should look like this before running:

```
your-project-folder/
├── app.py
├── secom.data          ✅ present
├── secom_labels.data   ✅ present
├── templates/
│   └── index.html
...
```

### 5. Run the app

```bash
python app.py
```

You will see this in the terminal when the model is ready:

```
Training Model... Please wait...
Model Trained and Ready!
* Running on http://127.0.0.1:5000
```

### 6. Open in browser

```
http://127.0.0.1:5000
```

---

## How to Use

1. Use the three sliders to set sensor deviation values (in sigma units)
2. Click **INITIATE SCAN**
3. The radar chart and log terminal update with the prediction result
4. If a defect is detected, click **EXPORT PDF REPORT** to download the full diagnostic report

---

## Running the SHAP Analysis Script (Optional)

`main.py` is a standalone script that trains the model, evaluates it, and prints the top 3 sensors causing failures using SHAP explainability. It does not start a web server.

```bash
python main.py
```

This will print a confusion matrix, failure catch rate, and a root cause prompt you can paste into any LLM for a detailed engineering report.

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server and API |
| `pandas` | Data loading and cleaning |
| `numpy` | Numerical operations |
| `scikit-learn` | Random Forest model |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `fpdf` | PDF report generation |
| `shap` | Model explainability (main.py only) |
| `matplotlib` | Plotting (main.py only) |

---

## Common Issues

**App starts but button stays on PROCESSING**
- The model failed to train. Check the terminal for errors.
- Most likely cause: `secom.data` or `secom_labels.data` is missing from the project root.

**`ModuleNotFoundError`**
- Run `pip install -r requirements.txt` again inside your virtual environment.
- Make sure your virtual environment is activated before running the app.

**Port already in use**
- Another process is using port 5000. Either stop that process or run Flask on a different port:
```bash
flask run --port 5001
```

**`secom.data` not found even though it's in the folder**
- Make sure you are running `python app.py` from inside the project folder, not from a different directory.

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn (Random Forest), imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Frontend:** Vanilla JS, Chart.js
- **Reports:** FPDF
- **Dataset:** SECOM — UCI Machine Learning Repository
=======
# Semi-AI-Virtual-Metrology
Semi-AI-Virtual-Metrology
>>>>>>> b92b0f07e5bec6a2b53008beff81b8faf860431f
