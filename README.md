# Heart Disease Prediction System — Web App

An interactive Streamlit UI for the trained heart disease prediction model.

## Files needed (all in the same folder)
- `app.py` — the Streamlit app
- `heart_disease_model.pkl` — trained model
- `scaler.pkl` — feature scaler
- `feature_names.pkl` — expected feature order
- `requirements.txt` — dependencies

## Setup & Run

```bash
# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Using the App
1. Fill in the patient's clinical details in the form (age, blood pressure, cholesterol, chest pain type, etc.).
2. Click **"🔍 Predict Heart Disease Risk"**.
3. View the result card, risk probability gauge, and patient summary.
4. Expand **"View raw input data sent to the model"** to see exactly what was fed to the model.

## Notes
- The model was trained on the UCI Cleveland Heart Disease dataset and is for **educational purposes only** — not a medical diagnostic tool.
- To deploy publicly, you can push this folder to GitHub and deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud).
