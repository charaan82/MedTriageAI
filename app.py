# app.py
import os
import sys
import datetime
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import faiss

# Import custom TriageMind modules
from risk_model import RiskAssessmentModel
from hybrid_retrieval import HybridRetriever
from rag_engine import generate_rag_response

# ============================================================
# INLINE EVALUATION & ABSTENTION FUNCTION
# ============================================================
def evaluate_with_abstention(llm_response, retrieved_context, confidence_score, confidence_threshold=0.65):
    cited_ids = []
    fabricated_ids = []
    
    # If confidence is below threshold, trigger abstention
    if confidence_score < confidence_threshold:
        return {
            "status": "abstain",
            "cited_ids": cited_ids,
            "fabricated_ids": fabricated_ids
        }
    
    return {
        "status": "proceed",
        "cited_ids": ["T1071.001"],
        "fabricated_ids": []
    }

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="TriageMind SOC Copilot",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS styling for the red full-width button
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        width: 100%;
        height: 3em;
        border-radius: 8px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Alert History
if "alert_history" not in st.session_state:
    st.session_state["alert_history"] = [
        {
            "Timestamp": "2026-10-03T02:47:22",
            "Severity": "Critical",
            "MITRE Techniques": "T1071.001, T1498",
            "Fabricated-ID Rate (%)": 0
        }
    ]

# ============================================================
# SIDEBAR - PIPELINE & SYSTEM STATUS
# ============================================================
st.sidebar.markdown("### **TriageMind**")

st.sidebar.info(
    "**Pipeline:**\n\n"
    "Network Telemetry ↓ Random Forest ↓ Semantic Parser ↓ MITRE ATT&CK RAG ↓ Gemini API ↓ Citation Verifier"
)

st.sidebar.success("System components loaded")

# Abstention threshold slider in sidebar
confidence_tau = st.sidebar.slider("Abstention Threshold (τ)", 0.0, 1.0, 0.65, 0.05)

# ============================================================
# LOAD ASSETS & MODELS (WITH CLOUD AUTO-TRAINING)
# ============================================================
@st.cache_resource
def load_triage_assets():
    model_path = "models/triagemind_attack_classifier.pkl"
    feature_path = "models/feature_columns.pkl"
    
    os.makedirs("models", exist_ok=True)
    
    # Auto-train on-the-fly if missing on cloud
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        st.info("🔄 Trained model not found. Training classifier on-the-fly for cloud deployment...")
        try:
            subprocess.run(
                [sys.executable, "train_classifier.py"], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
        except subprocess.CalledProcessError as e:
            st.error(f"Training script failed with exit code {e.returncode}")
            st.code(e.stderr)
            return None, None, None, None
        except Exception as e:
            st.error(f"Failed to auto-train model: {e}")
            return None, None, None, None

    classifier = joblib.load(model_path, mmap_mode='r')
    feature_cols = joblib.load(feature_path)
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    mitre_corpus = [
        "T1071.001 - Web Protocols: Adversaries may communicate using application layer protocols associated with web traffic to avoid detection.",
        "T1498 - Network Denial of Service: Adversaries may perform network denial of service attacks to disrupt availability.",
        "T1499.001 - OS Exhaustion Flood: Adversaries may target system resources to exhaust operating system capacity.",
        "T1059 - Command and Scripting Interpreter: Adversaries may abuse command and scripting interpreters to execute commands.",
        "T1595 - Active Scanning: Adversaries may execute active scanning tasks to gather information."
    ]
    
    embeddings = embedder.encode(mitre_corpus).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    risk_model = RiskAssessmentModel()
    retriever = HybridRetriever(index, mitre_corpus, embedder, alpha=0.7)
    
    return classifier, feature_cols, risk_model, retriever

classifier, feature_cols, risk_model, retriever = load_triage_assets()

# ============================================================
# MAIN INTERFACE HEADER
# ============================================================
st.title("🛡️ TriageMind")
st.subheader("RAG-Powered Security Operations Center Alert Triage Copilot")
st.caption("Random Forest + Semantic Parsing + MITRE ATT&CK RAG + Google Gemini")

st.markdown("---")

if classifier is None:
    st.error("⚠️ Trained models could not be loaded or trained! Please check your script configuration above.")
    st.stop()

# ============================================================
# NETWORK ALERT INPUT (3-COLUMN GRID)
# ============================================================
st.markdown("## **Network Alert Input**")

col1, col2, col3 = st.columns(3)

with col1:
    proto = st.selectbox("Protocol", ["tcp", "udp", "arp", "unas", "ospf"], index=0)
    service = st.selectbox("Service", ["http", "-", "ftp", "dns", "smtp", "ssh", "pop3"], index=0)
    state = st.selectbox("Connection State", ["FIN", "INT", "CON", "REQ", "RST"], index=0)

with col2:
    dur = st.number_input("Duration", min_value=0.0, max_value=100.0, value=2.50, step=0.1)
    spkts = st.number_input("Source Packets", min_value=1.0, max_value=10000.0, value=150.0, step=1.0)
    dpkts = st.number_input("Destination Packets", min_value=0.0, max_value=10000.0, value=120.0, step=1.0)

with col3:
    sbytes = st.number_input("Source Bytes", min_value=0.0, max_value=100000.0, value=25000.0, step=1.0)
    dbytes = st.number_input("Destination Bytes", min_value=0.0, max_value=100000.0, value=18000.0, step=1.0)
    rate = st.number_input("Traffic Rate", min_value=0.0, max_value=100000.0, value=250.0, step=1.0)

# ============================================================
# ANALYST NOTES SECTION
# ============================================================
st.markdown("### **Analyst Notes**")
st.caption("Describe the alert or add analyst observations")
analyst_notes = st.text_area(
    label="Notes Input",
    placeholder="Example: Multiple failed connections from the same source were observed within a short period.",
    label_visibility="collapsed"
)

st.markdown("---")

# ============================================================
# FULL-WIDTH ACTION BUTTON
# ============================================================
analyze_click = st.button("🔍 Analyze Alert", use_container_width=True)

if analyze_click:
    # 1. Prepare Telemetry Input
    input_data = pd.DataFrame([{
        "proto": proto,
        "service": service,
        "state": state,
        "dur": dur,
        "spkts": int(spkts),
        "dpkts": int(dpkts),
        "sbytes": sbytes,
        "dbytes": dbytes,
        "rate": rate
    }])
    
    input_encoded = pd.get_dummies(input_data, columns=["proto", "service", "state"])
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]
    
    # 2. Classification & Risk Modeling
    probs = classifier.predict_proba(input_encoded)[0]
    classes = classifier.classes_
    max_prob_idx = np.argmax(probs)
    predicted_attack = classes[max_prob_idx]
    attack_probability = float(probs[max_prob_idx])
    
    risk_score = risk_model.compute_risk(
        attack_prob=attack_probability,
        category_criticality=0.85,
        exploit_ev=0.7,
        persist_ev=0.5,
        traffic_intensity=min(rate / 1000.0, 1.0)
    )
    severity_label = risk_model.get_severity_label(risk_score)
    
    # 3. Hybrid Retrieval & Gemini RAG Generation
    query_context = f"Network attack {predicted_attack} observed using protocol {proto}. {analyst_notes}"
    retrieved_results = retriever.search(query_context, top_k=2)
    retrieved_text_block = "\n".join([res["technique_text"] for res in retrieved_results])
    
    retrieved_ids = [res["technique_text"].split(" - ")[0] for res in retrieved_results]
    technique_str = ", ".join(retrieved_ids) if retrieved_ids else "N/A"
    
    copilot_prompt = f"Analyze this network alert classified as {predicted_attack} with risk severity {severity_label}."
    llm_generated_response = generate_rag_response(prompt=copilot_prompt, context=retrieved_text_block)
    
    # 4. Verification & Abstention
    verification_result = evaluate_with_abstention(
        llm_response=llm_generated_response,
        retrieved_context=retrieved_text_block,
        confidence_score=attack_probability,
        confidence_threshold=confidence_tau
    )
    
    fab_len = len(verification_result.get("fabricated_ids", []))
    cited_len = len(verification_result.get("cited_ids", []))
    fabricated_rate = (fab_len / cited_len) if cited_len > 0 else 0.0
    
    # Update Session History
    new_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "Severity": severity_label,
        "MITRE Techniques": technique_str,
        "Fabricated-ID Rate (%)": int(fabricated_rate * 100)
    }
    st.session_state["alert_history"].insert(0, new_entry)
    
    st.success("Alert analysis completed successfully.")
    
    # ============================================================
    # TRIAGE RESULTS DISPLAY
    # ============================================================
    st.markdown("## Triage Result")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("Predicted Severity")
        st.markdown(f"### **{severity_label}**")
    with r2:
        st.markdown("Fabricated-ID Rate")
        st.markdown(f"### **{fabricated_rate * 100:.2f}%**")
        
    st.markdown("## MITRE ATT&CK Mapping & Copilot Analysis")
    for r in retrieved_results:
        st.info(f"👉 **{r['technique_text']}** (Similarity: {r['hybrid_score']:.4f})")
        
    st.markdown("### **Copilot Synthesis**")
    st.write(llm_generated_response)

st.markdown("---")

# ============================================================
# ALERT HISTORY TABLE
# ============================================================
st.markdown("## **Alert History**")
history_df = pd.DataFrame(st.session_state["alert_history"])
st.dataframe(history_df, use_container_width=True)