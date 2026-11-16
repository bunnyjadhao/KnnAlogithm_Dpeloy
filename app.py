"""
Heart Disease Risk Predictor - Flask app for Render
Model: KNeighborsClassifier (knnmodel.pkl)

Features (in model order) and the numeric encoding used during training
(sklearn LabelEncoder = alphabetical order):
    Sex            : F=0, M=1
    ChestPainType  : ASY=0, ATA=1, NAP=2, TA=3
    RestingECG     : LVH=0, Normal=1, ST=2
    ExerciseAngina : N=0, Y=1
    ST_Slope       : Down=0, Flat=1, Up=2
Target: 0 = Normal, 1 = Heart Disease
"""

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "knnmodel.pkl"))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# ----------------------------------------------------------------------------
# Form definition (drives both the HTML form and the validation)
# ----------------------------------------------------------------------------
FIELDS = [
    {"name": "Age", "label": "Age", "icon": "🎂", "type": "number",
     "min": 1, "max": 120, "step": 1, "placeholder": "e.g. 54", "unit": "years"},
    {"name": "Sex", "label": "Sex", "icon": "🧑", "type": "select",
     "options": [("1", "Male"), ("0", "Female")]},
    {"name": "ChestPainType", "label": "Chest Pain Type", "icon": "💔", "type": "select",
     "options": [("0", "Asymptomatic (ASY)"), ("1", "Atypical Angina (ATA)"),
                 ("2", "Non-Anginal Pain (NAP)"), ("3", "Typical Angina (TA)")]},
    {"name": "RestingBP", "label": "Resting Blood Pressure", "icon": "🩸", "type": "number",
     "min": 0, "max": 300, "step": 1, "placeholder": "e.g. 130", "unit": "mm Hg"},
    {"name": "Cholesterol", "label": "Cholesterol", "icon": "🧪", "type": "number",
     "min": 0, "max": 700, "step": 1, "placeholder": "e.g. 220", "unit": "mg/dL"},
    {"name": "FastingBS", "label": "Fasting Blood Sugar", "icon": "🍬", "type": "select",
     "options": [("0", "Below 120 mg/dL"), ("1", "Above 120 mg/dL")]},
    {"name": "RestingECG", "label": "Resting ECG", "icon": "📈", "type": "select",
     "options": [("1", "Normal"), ("2", "ST-T Wave Abnormality (ST)"),
                 ("0", "Left Ventricular Hypertrophy (LVH)")]},
    {"name": "MaxHR", "label": "Maximum Heart Rate", "icon": "💓", "type": "number",
     "min": 40, "max": 250, "step": 1, "placeholder": "e.g. 150", "unit": "bpm"},
    {"name": "ExerciseAngina", "label": "Exercise-Induced Angina", "icon": "🏃", "type": "select",
     "options": [("0", "No"), ("1", "Yes")]},
    {"name": "Oldpeak", "label": "Oldpeak (ST Depression)", "icon": "📉", "type": "number",
     "min": -3, "max": 7, "step": 0.1, "placeholder": "e.g. 1.2", "unit": ""},
    {"name": "ST_Slope", "label": "ST Slope", "icon": "⛰️", "type": "select",
     "options": [("2", "Upsloping (Up)"), ("1", "Flat"), ("0", "Downsloping (Down)")]},
]


def parse_and_validate(data):
    """Return (values_dict, errors_list). Values are floats ready for the model."""
    values, errors = {}, []
    for field in FIELDS:
        name = field["name"]
        raw = data.get(name, "")
        if raw is None or str(raw).strip() == "":
            errors.append(f"{field['label']} is required.")
            continue
        try:
            num = float(raw)
        except ValueError:
            errors.append(f"{field['label']} must be a number.")
            continue

        if field["type"] == "select":
            allowed = {float(v) for v, _ in field["options"]}
            if num not in allowed:
                errors.append(f"{field['label']} has an invalid option.")
                continue
        else:
            if not (field["min"] <= num <= field["max"]):
                errors.append(
                    f"{field['label']} must be between {field['min']} and {field['max']}."
                )
                continue
        values[name] = num
    return values, errors


def run_prediction(values):
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    risk = float(proba[classes.index(1)]) * 100  # probability of heart disease
    return pred, round(risk, 1)


# ----------------------------------------------------------------------------
# Template
# ----------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heart Disease Risk Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --text: #f4f1ff;
    --muted: #b9b4d8;
    --glass: rgba(255, 255, 255, 0.08);
    --glass-border: rgba(255, 255, 255, 0.18);
    --grad-main: linear-gradient(90deg, #00c6ff 0%, #7b2ff7 55%, #ff2d95 100%);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    color: var(--text);
    min-height: 100vh;
    padding: 40px 16px 60px;
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #3a1c71, #1a2a6c);
    background-size: 400% 400%;
    animation: bgShift 18s ease infinite;
    position: relative;
    overflow-x: hidden;
  }
  @keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* floating colour blobs */
  .blob { position: fixed; border-radius: 50%; filter: blur(90px); opacity: 0.45; z-index: 0; pointer-events: none; }
  .blob.b1 { width: 380px; height: 380px; background: #ff2d95; top: -120px; left: -100px; }
  .blob.b2 { width: 420px; height: 420px; background: #00c6ff; bottom: -160px; right: -120px; }
  .blob.b3 { width: 280px; height: 280px; background: #7b2ff7; top: 45%; left: 55%; }

  .container { position: relative; z-index: 1; max-width: 920px; margin: 0 auto; }

  header { text-align: center; margin-bottom: 30px; }
  .heart { font-size: 54px; display: inline-block; animation: beat 1.3s infinite; }
  @keyframes beat { 0%,100% { transform: scale(1); } 15% { transform: scale(1.18); } 30% { transform: scale(1); } 45% { transform: scale(1.12); } }
  h1 {
    font-size: clamp(1.7rem, 4vw, 2.6rem);
    font-weight: 700;
    margin-top: 6px;
    background: var(--grad-main);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  header p { color: var(--muted); font-weight: 300; margin-top: 8px; }

  .card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 32px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
  }

  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px 24px; }
  @media (max-width: 680px) { .grid { grid-template-columns: 1fr; } .card { padding: 22px; } }

  .field label { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 7px; color: #dcd7ff; }
  .field label .unit { color: var(--muted); font-weight: 300; margin-left: 4px; }

  .field input, .field select {
    width: 100%;
    padding: 13px 14px;
    font-family: inherit;
    font-size: 0.95rem;
    color: var(--text);
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 12px;
    outline: none;
    transition: border-color .2s, box-shadow .2s, background .2s;
  }
  .field input::placeholder { color: rgba(255,255,255,0.35); }
  .field select option { color: #1b1740; background: #fff; }
  .field input:focus, .field select:focus {
    border-color: #00c6ff;
    background: rgba(255, 255, 255, 0.12);
    box-shadow: 0 0 0 4px rgba(0, 198, 255, 0.18);
  }

  .btn {
    display: block; width: 100%; margin-top: 30px;
    padding: 16px; border: none; border-radius: 14px; cursor: pointer;
    font-family: inherit; font-size: 1.05rem; font-weight: 600; letter-spacing: .4px; color: #fff;
    background: var(--grad-main); background-size: 200% 100%;
    box-shadow: 0 12px 30px rgba(123, 47, 247, 0.45);
    transition: transform .2s, background-position .5s, box-shadow .2s;
  }
  .btn:hover { transform: translateY(-2px); background-position: 100% 0; box-shadow: 0 16px 38px rgba(255, 45, 149, 0.45); }
  .btn:disabled { opacity: .7; cursor: wait; }

  .errors {
    background: rgba(255, 65, 108, 0.15); border: 1px solid rgba(255, 65, 108, 0.5);
    border-radius: 14px; padding: 14px 18px; margin-bottom: 22px; font-size: 0.9rem;
  }
  .errors li { margin-left: 18px; }

  /* result */
  .result { margin-top: 28px; border-radius: 22px; padding: 28px; text-align: center; color: #fff; animation: pop .5s ease; }
  @keyframes pop { from { opacity: 0; transform: translateY(14px) scale(.97); } to { opacity: 1; transform: none; } }
  .result.high { background: linear-gradient(135deg, #ff416c, #ff4b2b); box-shadow: 0 20px 45px rgba(255, 65, 108, 0.4); }
  .result.low  { background: linear-gradient(135deg, #11998e, #38ef7d); box-shadow: 0 20px 45px rgba(56, 239, 125, 0.3); }
  .result .emoji { font-size: 46px; }
  .result h2 { font-size: 1.5rem; margin: 8px 0 4px; }
  .result p { font-weight: 300; opacity: .95; }
  .meter { height: 14px; border-radius: 999px; background: rgba(0,0,0,.25); margin: 20px auto 6px; max-width: 460px; overflow: hidden; }
  .meter span { display: block; height: 100%; border-radius: 999px; background: rgba(255,255,255,.92); transition: width 1s ease; }
  .meter-label { font-size: .9rem; font-weight: 500; }

  .disclaimer { margin-top: 26px; text-align: center; font-size: .78rem; color: var(--muted); line-height: 1.6; }
  footer { text-align: center; margin-top: 28px; font-size: .8rem; color: var(--muted); }
</style>
</head>
<body>
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>

  <div class="container">
    <header>
      <div class="heart">❤️</div>
      <h1>Heart Disease Risk Predictor</h1>
      <p>Enter the patient details below and let the KNN model estimate the risk.</p>
    </header>

    <div class="card">
      {% if errors %}
      <div class="errors"><ul>{% for e in errors %}<li>{{ e }}</li>{% endfor %}</ul></div>
      {% endif %}

      <form method="POST" action="/" id="predict-form">
        <div class="grid">
          {% for f in fields %}
          <div class="field">
            <label for="{{ f.name }}">{{ f.icon }} {{ f.label }}{% if f.unit %}<span class="unit">({{ f.unit }})</span>{% endif %}</label>
            {% if f.type == 'select' %}
              <select id="{{ f.name }}" name="{{ f.name }}" required>
                {% for val, text in f.options %}
                <option value="{{ val }}" {% if form.get(f.name) == val %}selected{% endif %}>{{ text }}</option>
                {% endfor %}
              </select>
            {% else %}
              <input type="number" id="{{ f.name }}" name="{{ f.name }}"
                     min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                     placeholder="{{ f.placeholder }}" value="{{ form.get(f.name, '') }}" required>
            {% endif %}
          </div>
          {% endfor %}
        </div>
        <button class="btn" type="submit" id="submit-btn">🔍 Predict Now</button>
      </form>

      {% if result %}
      <div class="result {{ 'high' if result.prediction == 1 else 'low' }}" id="result">
        <div class="emoji">{{ '⚠️' if result.prediction == 1 else '✅' }}</div>
        <h2>{{ 'High Risk of Heart Disease' if result.prediction == 1 else 'Low Risk - Likely Normal' }}</h2>
        <p>{{ 'The model predicts signs of heart disease. Please consult a cardiologist.' if result.prediction == 1 else 'The model does not detect signs of heart disease for these inputs.' }}</p>
        <div class="meter"><span style="width: {{ result.risk }}%"></span></div>
        <div class="meter-label">Heart disease probability: {{ result.risk }}%</div>
      </div>
      {% endif %}

      <p class="disclaimer">
        This tool is for educational purposes only and is not a medical diagnosis.<br>
        Always consult a qualified healthcare professional for medical advice.
      </p>
    </div>

    <footer>Built with Flask &amp; scikit-learn &middot; Deployed on Render</footer>
  </div>

<script>
  document.getElementById('predict-form').addEventListener('submit', function () {
    var b = document.getElementById('submit-btn');
    b.disabled = true; b.textContent = 'Analyzing...';
  });
  var r = document.getElementById('result');
  if (r) { r.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    form, errors, result = {}, [], None
    if request.method == "POST":
        form = request.form.to_dict()
        values, errors = parse_and_validate(form)
        if not errors:
            pred, risk = run_prediction(values)
            result = {"prediction": pred, "risk": risk}
    return render_template_string(PAGE, fields=FIELDS, form=form, errors=errors, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API. Send numeric codes, e.g. {"Age":54,"Sex":1,"ChestPainType":0,...}"""
    data = request.get_json(silent=True) or {}
    values, errors = parse_and_validate(data)
    if errors:
        return jsonify({"errors": errors}), 400
    pred, risk = run_prediction(values)
    return jsonify({
        "prediction": pred,
        "label": "Heart Disease" if pred == 1 else "Normal",
        "heart_disease_probability_percent": risk,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
