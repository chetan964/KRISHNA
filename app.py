import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import matplotlib.pyplot as plt
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# PDF text extraction
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text

# Extract simple keywords
def extract_keywords(text):
    words = [w.lower() for w in text.split() if len(w) > 4]
    return list(set(words))

# UI
st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("📄 AI Resume Ranker & Job Match Visualizer")

# ✅ New: Role input
role_title = st.text_input("🎯 Job Role Title (e.g., Data Analyst, Software Engineer)")

job_desc = st.text_area("📝 Paste Job Description", height=150)
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])

# Run button
if st.button("🚀 Analyze & Visualize"):
    if not role_title:
        st.error("Please enter the job role.")
    elif not job_desc or not uploaded_files:
        st.error("Please provide job description and at least one resume.")
    else:
        jd_embedding = model.encode(job_desc, convert_to_tensor=True)
        jd_keywords = extract_keywords(job_desc)

        results = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            res_embedding = model.encode(text, convert_to_tensor=True)

            # Match Score (cosine → percentage)
            sim = util.cos_sim(jd_embedding, res_embedding).item()
            match_percent = round(sim * 100, 2)

            # Skill analysis
            resume_keywords = extract_keywords(text)
            found = [kw for kw in jd_keywords if kw in resume_keywords]
            missing = [kw for kw in jd_keywords if kw not in resume_keywords]

            coverage = round((len(found) / len(jd_keywords)) * 100, 2) if jd_keywords else 0

            results.append({
                "file": file.name,
                "match": match_percent,
                "coverage": coverage,
                "missing_pct": 100 - coverage,
                "found": found,
                "missing": missing,
            })

        # Convert to DataFrame for graph
        df = pd.DataFrame({
            "Resume": [r["file"] for r in results],
            "Match %": [r["match"] for r in results],
            "Skill Coverage %": [r["coverage"] for r in results],
            "Missing %": [r["missing_pct"] for r in results],
        })

        # ✅ Clean single graph
        st.subheader("📊 Resume Match & Skill Coverage Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        bar_width = 0.25
        x = range(len(df))

        ax.bar([i - bar_width for i in x], df["Match %"], width=bar_width)
        ax.bar(x, df["Skill Coverage %"], width=bar_width)
        ax.bar([i + bar_width for i in x], df["Missing %"], width=bar_width)

        ax.set_xticks(x)
        ax.set_xticklabels(df["Resume"], rotation=45, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.legend(["Match %", "Skill Coverage %", "Missing %"])

        st.pyplot(fig)

        # ✅ Clean minimal analysis section
        st.subheader("📌 Resume Analysis (Clean Output)")

        for r in results:
            st.markdown(f"### ✅ {r['file']} — {r['match']}% Match")
            st.write("**Skills Found:**", ", ".join(r["found"]) if r["found"] else "None")
            st.write("**Skills Missing:**", ", ".join(r["missing"]) if r["missing"] else "None")
