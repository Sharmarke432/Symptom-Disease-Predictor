import streamlit as st
import joblib

# ---------- 1. Load model and vectorizer ----------
st.set_page_config(
    page_title="Symptom-based Disease Suggestion",
    page_icon="🩺",
    layout="centered",
)

@st.cache_resource
def load_model():
    vec = joblib.load("text_vec.pkl")           # TF-IDF vectorizer
    clf = joblib.load("text_disease_model.pkl") # LogisticRegression model
    return vec, clf

vec, clf = load_model()

# ---------- 2. Helper: predict diseases from free text ----------

def predict_diseases_text_only(text, vec, clf, top_k=5):
    X_vec = vec.transform([text])
    probs = clf.predict_proba(X_vec)[0]
    classes = clf.classes_
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

# ---------- 3. Streamlit UI ----------

st.title("Symptom-based Disease Suggestion (Demo)")
st.write(
    "Type your symptoms in natural language and get a list of possible diseases "
    "with model similarity scores (confidence percentages). "
    "**This is not medical advice. For any health concerns, consult a doctor.**"
)

user_text = st.text_area(
    "Describe your symptoms:",
    height=150,
    placeholder="Example: I've had a runny nose, dry throat and a fever for two days..."
)

top_k = st.slider("Number of suggestions to show:", min_value=3, max_value=10, value=5, step=1)

if st.button("Check possible diseases"):
    if not user_text.strip():
        st.warning("Please type a description of your symptoms first.")
    else:
        results = predict_diseases_text_only(user_text, vec, clf, top_k=top_k)

        st.subheader("Possible diseases (model similarity)")
        for disease, p in results:
            st.write(f"**{disease}**: {p:.1%} similarity")
