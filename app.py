from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GEMINI_API_KEY"] = "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"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import streamlit as st
import pickle
import requests

# ── CONFIG ──
DATASET_REAL = "dataset/real"
DATASET_FAKE = "dataset/fake"
MODEL_PATH = "model.pkl"

# ── FEATURE EXTRACTION ──
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))

    # 1. OpenCV — Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (224 * 224)

    # 2. OpenCV — Blur detection
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 3. Color features
    mean_colors = cv2.mean(img)[:3]

    # 4. Noise level
    noise = np.std(gray)

    # 5. Contrast
    contrast = gray.max() - gray.min()

    # 6. Histogram features
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist.flatten() / hist.sum()

    features = [
        edge_density,
        blur_score,
        *mean_colors,
        noise,
        contrast,
        *hist
    ]
    return features

# ── TRAIN MODEL ──
def train_model():
    X, y = [], []

    # Real images
    for f in os.listdir(DATASET_REAL):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_features(os.path.join(DATASET_REAL, f))
            if feat:
                X.append(feat)
                y.append(1)  # 1 = Real

    # Fake images
    for f in os.listdir(DATASET_FAKE):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_features(os.path.join(DATASET_FAKE, f))
            if feat:
                X.append(feat)
                y.append(0)  # 0 = Fake

    X = np.array(X)
    y = np.array(y)

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model, acc

# ── LOAD MODEL ──
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# ── PREDICT ──
def predict(img_path, model):
    feat = extract_features(img_path)
    if feat is None:
        return None, 0
    feat = np.array(feat).reshape(1, -1)
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]
    score = int(prob[1] * 100) if pred == 1 else int(prob[0] * 100)
    return "AUTHENTIC" if pred == 1 else "FAKE", score

# ── OCR ──
def extract_text(img_path):
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except:
        return "OCR not available"

# ── GEMINI EXPLANATION ──
def get_explanation(status, score, ocr_text, api_key):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        prompt = f"Document Analysis: Status={status}, Score={score}%, OCR={ocr_text[:200]}. Explain why {status}, key observations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": "openrouter/free", "messages": [{"role": "user", "content": prompt}]}
        
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        
        # ✅ ఇది add చేయండి — debug కోసం
        print("API Response:", result)  
        
        # ✅ safe గా access చేయండి
        if "choices" in result:
            return result['choices'][0]['message']['content']
        elif "error" in result:
            return f"API Error: {result['error']['message']}"
        else:
            return f"Unexpected response: {result}"
            
    except Exception as e:
        return f"Gemini Error: {e}"                                                          
# ── STREAMLIT UI ──
st.set_page_config(
    page_title="AI Document Authenticity Checker",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
    .main { background-color: #0A0F1E; }
    .stApp { background: linear-gradient(135deg, #0A0F1E, #0D1B2E); }
    h1 { color: #00D4FF !important; }
    .metric-card {
        background: rgba(26,34,53,0.8);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 AI Document Authenticity Checker")
st.markdown("**OCR + OpenCV + AI powered document forensics**")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = os.getenv("GEMINI_API_KEY") or st.text_input("🔑 Gemini API Key", type="password")

    language = st.selectbox("🌐 Language", ["English", "Telugu", "Hindi"])
    st.divider()
    st.header("🤖 Model")
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            model, acc = train_model()
            st.session_state['model'] = model
            st.success(f"✅ Model trained! Accuracy: {acc*100:.1f}%")

    if st.button("Load Existing Model"):
        model = load_model()
        if model:
            st.session_state['model'] = model
            st.success("✅ Model loaded!")
        else:
            st.warning("⚠️ No model found. Train first!")

# Camera
st.subheader("📷 Camera Capture")
img_file = st.camera_input("Take a photo of document")
if img_file:
    tmp_path = "temp_camera.jpg"
    with open(tmp_path, 'wb') as f:
        f.write(img_file.getvalue())
    st.success("✅ Photo captured!")
# Main
uploaded = st.file_uploader(
    "📄 Upload Document",
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    # Save temp file
    tmp_path = f"temp_{uploaded.name}"
    with open(tmp_path, 'wb') as f:
        f.write(uploaded.getvalue())

    st.image(uploaded, caption="Uploaded Document", use_column_width=True)

    if st.button("🔍 Analyze Document", type="primary"):
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train or load model first!")
        else:
            with st.spinner("Analyzing..."):
                model = st.session_state['model']

                # Predict
                status, score = predict(tmp_path, model)

                # OCR
                ocr_text = extract_text(tmp_path)

                # AI Explanation
                explanation = ""
                if api_key:
                    explanation = get_explanation(
                        status, score, ocr_text, api_key
                    )

            # Results
            st.divider()
            st.subheader("📊 Analysis Results")

            col1, col2 = st.columns(2)
            with col1:
                if status == "AUTHENTIC":
                    st.success(f"✅ {status}")
                else:
                    st.error(f"❌ {status}")
            with col2:
                st.metric("Authenticity Score", f"{score}%")

            st.progress(score / 100)
            st.session_state['last_result'] = {'status': status, 'score': score}

            # OCR Results
            st.subheader("📝 OCR — Extracted Text")
            st.code(ocr_text if ocr_text else "No text detected")

            # AI Explanation
            if explanation:
                st.subheader("🧠 AI Forensic Analysis")
                st.info(explanation)

    # PDF Download
if 'last_result' in st.session_state:
    result = st.session_state['last_result']
    if st.button("📄 Download PDF Report"):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Document Authenticity Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Status: {result['status']}", ln=True)
        pdf.cell(200, 10, txt=f"Score: {result['score']}%", ln=True)
        pdf.output("report.pdf")
        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, "report.pdf")
        # History
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_result' in st.session_state:
    result = st.session_state['last_result']
    st.session_state['history'].append(result)
    st.subheader("🗂️ History")
    for i, item in enumerate(st.session_state['history']):
        st.write(f"{i+1}. Status: {item['status']} | Score: {item['score']}%")    
    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)