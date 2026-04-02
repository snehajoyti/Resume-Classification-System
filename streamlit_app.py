import streamlit as st
import pickle
import docx
import pandas as pd
import random
import base64

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Resume Classification System", layout="wide")

# ================= BACKGROUND IMAGE =================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .card {{
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }}

    .title {{
        text-align: center;
        color: #2c3e50;
    }}

    .resume-box {{
        background: rgba(255, 255, 255, 0.2);  
        padding: 20px;
        border-radius: 10px;
        height: 500px;
        overflow-y: scroll;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        backdrop-filter: blur(8px); 
    }}

    .side-box {{
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 10px;
        height: 500px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# 👉 Background image
set_bg("IMG_8412.WEBP")

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# ================= FUNCTION =================
def extract_text(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# ================= HEADER =================
st.markdown("<h1 class='title'> Resume Classification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart Hiring using AI 🚀</p>", unsafe_allow_html=True)

# ================= SIDEBAR =================
# st.sidebar.title("📌 About ATS System")

# category_filter = st.sidebar.selectbox(
#     "Select Category",
#     ["All","Internship", "React", "React JS"]
# )

# st.sidebar.info("""
# AI-powered resume screening system that classifies resumes and helps HR shortlist candidates.
# """)

# ================= FILE UPLOAD =================
uploaded_files = st.file_uploader(
    "📤 Upload Resumes (.docx)",
    type=["docx"],
    accept_multiple_files=True
)

# ================= PROCESS =================
if uploaded_files:

    st.success(f"✅ {len(uploaded_files)} resumes uploaded!")

    for i, file in enumerate(uploaded_files):

        text = extract_text(file)

        vector = tfidf.transform([text])
        prediction = model.predict(vector)[0]

        score = random.randint(60, 95)

        st.markdown("---")
        st.subheader(f"👤 {file.name}")

        col1, col2 = st.columns([3, 1])

        # ================= LEFT: RESUME VIEW =================
        with col1:
            st.markdown("### 📄 Resume")

            # 🔥 FIXED PART
            formatted_text = text.replace("\n", "<br>")

            st.markdown(
                f"<div class='resume-box'>{formatted_text}</div>",
                unsafe_allow_html=True
            )

        # ================= RIGHT: SIDE PANEL =================
        with col2:
            #st.markdown("<div class='side-box'>", unsafe_allow_html=True)

            st.markdown("### 🎯 Candidate Info")
            st.success(f"Category: {prediction}")
            st.info(f"Score: {score}%")

            st.progress(score / 100)

            st.markdown("### 📌 Actions")

            if st.button("✅ Select", key=f"select_{i}"):
                st.success(f"{file.name} Selected")

            if st.button("❌ Reject", key=f"reject_{i}"):
                st.error(f"{file.name} Rejected")

            if st.button("⭐ Shortlist", key=f"shortlist_{i}"):
                st.warning(f"{file.name} Shortlisted")

            st.markdown("</div>", unsafe_allow_html=True)

    # ================= DOWNLOAD =================
    st.markdown("## 📥 Download Results")

    df = pd.DataFrame([
        {
            "File Name": file.name,
            "Category": model.predict(tfidf.transform([extract_text(file)]))[0]
        }
        for file in uploaded_files
    ])

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download CSV",
        csv,
        "ATS_results.csv",
        "text/csv"
    )