Symptom-based Disease Suggestion (Demo)
This is a simple Streamlit web app that suggests likely diseases from a free‑text description of symptoms, using a TF‑IDF + Logistic Regression model and a symptom–disease matrix as auxiliary data.

Features
Enter symptoms in plain English (e.g. “runny nose, dry throat, fever for two days”).

Get a ranked list of likely diseases with similarity/confidence percentages.

Uses a trained text model (TfidfVectorizer + LogisticRegression).
​

Uses a cleaned binary symptom matrix (DiseaseAndSymptoms.csv) where all‑zero symptom columns are removed.
​

Runs entirely in the browser via Streamlit!

.
├── app.py                      # Streamlit app
├── DiseaseAndSymptoms.csv      # Disease ↔ symptoms matrix
├── text_vec.pkl                # Trained TfidfVectorizer
├── text_disease_model.pkl      # Trained LogisticRegression model
└── requirements.txt            # Python dependencies


Installation Steps:

python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt #contains the dependencies

Model Files
The app expects two pre‑trained objects saved with joblib:

  text_vec.pkl – a fitted TfidfVectorizer trained on symptom text.

  text_disease_model.pkl – a fitted LogisticRegression (or similar classifier) trained to predict diseases from TF‑IDF features.
​
streamlit run app.py #run locally
