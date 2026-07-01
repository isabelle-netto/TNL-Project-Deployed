import streamlit as st
import joblib
import re
from pathlib import Path

# -------------------------------
# Page setup
# -------------------------------

st.set_page_config(
    page_title="Malaysia Tech Sentiment Analyzer",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
        padding: 45px;
        border-radius: 22px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.92;
    }

    .info-card {
        background-color: white;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
        border-left: 6px solid #2563eb;
        margin-bottom: 20px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #1e3a8a;
    }

    .metric-label {
        font-size: 14px;
        color: #64748b;
    }

    .result-positive {
        background-color: #dcfce7;
        color: #166534;
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid #22c55e;
        font-size: 22px;
        font-weight: 700;
    }

    .result-negative {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid #ef4444;
        font-size: 22px;
        font-weight: 700;
    }

    .result-neutral {
        background-color: #e0f2fe;
        color: #075985;
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid #0ea5e9;
        font-size: 22px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# File paths
# -------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "Normal Models" / "models" / "best_model.pkl"
VECTORIZER_PATH = BASE_DIR / "Normal Models" / "TF-IDF file" / "best_tfidf_vectorizer.pkl"

# -------------------------------
# Load trained prediction system
# -------------------------------

@st.cache_resource
def load_prediction_system():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_prediction_system()

# -------------------------------
# Text cleaning
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def format_label(prediction):
    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
        "0": "Negative",
        "1": "Neutral",
        "2": "Positive",
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "Negative": "Negative",
        "Neutral": "Neutral",
        "Positive": "Positive"
    }
    return label_map.get(prediction, str(prediction))

# -------------------------------
# Simple aspect detection
# -------------------------------

def detect_aspect(text):
    text = text.lower()

    aspects = {
        "Environment / Sustainability": [
            "environment", "water", "electricity", "energy", "pollution",
            "carbon", "sustainability", "climate"
        ],
        "Jobs / Employment": [
            "job", "jobs", "career", "employment", "retrenched",
            "replace", "worker", "salary"
        ],
        "AI Development": [
            "ai", "artificial intelligence", "automation", "robot",
            "technology", "innovation"
        ],
        "Data Centre": [
            "data centre", "data center", "cloud", "server",
            "computing", "digital hub"
        ],
        "Economy / Investment": [
            "investment", "economy", "growth", "business",
            "industry", "company", "market"
        ],
        "Public Concern": [
            "worried", "concern", "danger", "risk", "bad",
            "problem", "issue"
        ]
    }

    for aspect, keywords in aspects.items():
        for keyword in keywords:
            if keyword in text:
                return aspect

    return "General Technology Opinion"

# -------------------------------
# Hero section
# -------------------------------

st.markdown("""
<div class="hero">
    <h1>Malaysia Tech Sentiment Analyzer</h1>
    <p>
        Analyse public opinions towards AI technology, data centres, and Malaysia’s growing digital landscape.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Metric cards
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">2,122</div>
        <div class="metric-label">Collected YouTube Comments</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">3</div>
        <div class="metric-label">Sentiment Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">AI</div>
        <div class="metric-label">Powered Sentiment Prediction</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# Main prediction section
# -------------------------------

left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("""
    <div class="info-card">
        <h3>Analyse a Comment</h3>
        <p>
            Enter a public comment related to AI, data centres, jobs, investment, 
            or technology development in Malaysia.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sample = st.selectbox(
        "Choose a sample comment or type your own below:",
        [
            "",
            "AI data centres in Malaysia will create more job opportunities.",
            "Data centres are bad for the environment.",
            "Malaysia launched a new AI office.",
            "This technology development is good for the country's economy.",
            "AI may replace many workers in Malaysia."
        ]
    )

    user_input = st.text_area(
        "Enter comment:",
        value=sample,
        height=140,
        placeholder="Example: AI data centres in Malaysia will create more job opportunities."
    )

    analyse_button = st.button("Analyse Sentiment", use_container_width=True)

with right_col:
    st.markdown("""
    <div class="info-card">
        <h3>What This System Shows</h3>
        <p>The system identifies:</p>
        <ul>
            <li>Whether the comment is positive, neutral, or negative</li>
            <li>The main topic or concern in the comment</li>
            <li>A cleaner version of the text for processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Prediction result
# -------------------------------

if analyse_button:
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        sentiment = format_label(prediction)
        aspect = detect_aspect(user_input)

        st.markdown("## Prediction Result")

        if sentiment == "Positive":
            st.markdown(f"""
            <div class="result-positive">
                ✅ Sentiment: {sentiment}
            </div>
            """, unsafe_allow_html=True)

        elif sentiment == "Negative":
            st.markdown(f"""
            <div class="result-negative">
                ❌ Sentiment: {sentiment}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-neutral">
                Sentiment: {sentiment}
            </div>
            """, unsafe_allow_html=True)

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Detected Aspect", aspect)

        with result_col2:
            st.metric("Analysis Status", "Completed")

        with st.expander("View text processing details"):
            st.write("**Original Text:**", user_input)
            st.write("**Pre-processed Text Used by System:**", cleaned_text)
            st.caption(
                "The system converts text to lowercase and removes unnecessary symbols, "
                "links, usernames, hashtags, and extra spaces before prediction."
            )

# -------------------------------
# About section
# -------------------------------

st.markdown("---")

st.markdown("""
<div class="info-card">
    <h3>About This System</h3>
    <p>
        This sentiment analysis system was developed for the Malaysia Technology Scene domain. 
        It analyses public opinions towards AI technology and data centres in Malaysia using 
        Natural Language Processing techniques.
    </p>
    <p>
        The system classifies comments into three sentiment categories: positive, neutral, 
        and negative. It also identifies the main topic area of the comment, such as environment, 
        jobs, investment, AI development, or data centres.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    TNL6323 Natural Language Processing Project | Malaysia Tech Sentiment Analyzer
</div>
""", unsafe_allow_html=True)