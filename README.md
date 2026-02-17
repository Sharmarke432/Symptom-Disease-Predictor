# Symptom-based Disease Suggestion

A simple Streamlit web app that suggests likely diseases from a free‑text description of symptoms, using a TF‑IDF + Logistic Regression text model and a disease–symptom matrix.[web:135][web:139]

> ⚠️ Educational demo only. This app does **not** provide medical advice. For any health concerns, always consult a qualified healthcare professional.

---

## Features

- Enter symptoms in natural language (e.g. “runny nose, dry throat, and a fever for two days”).  
- Get a ranked list of likely diseases with similarity/confidence percentages.  
- Uses a trained `TfidfVectorizer` + `LogisticRegression` pipeline for text classification.[web:135]  
- Uses a cleaned binary disease–symptom matrix where all‑zero symptom columns are removed.[web:139]  
- Runs as a lightweight Streamlit app deployable on Streamlit Community Cloud.[web:148][web:149]

---

## Project Structure

```text
.
├── app.py                      # Streamlit app (main entry point)
├── DiseaseAndSymptoms.csv      # Disease ↔ symptom matrix
├── text_vec.pkl                # Trained TfidfVectorizer
├── text_disease_model.pkl      # Trained LogisticRegression model
└── requirements.txt            # Python dependencies
```

## Clone the repo
```
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

## (Optional but recommended) Create and activate a virtual environment
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Run locally
```
streamlit run app.py
```
