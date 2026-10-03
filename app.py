import os
import re
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from semantic_parser import parse_alert
from rag_engine import retrieve
from ai_engine import generate_analysis
from verifier import verify_citations


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "models/triagemind_classifier.pkl"
FEATURE_FILE = "models/feature_columns.pkl"

DB_FILE = "database/triagemind_history.db"


FEATURES = [
    "proto",
    "service",
    "state",
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate"
]


CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state"
]


NUMERIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate"
]


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    feature_columns = joblib.load(
        FEATURE_FILE
    )

    return model, feature_columns


model, feature_columns = load_model()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    os.makedirs(
        "database",
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            severity TEXT,
            mitre_ids TEXT,
            fabricated_rate REAL,
            alert_text TEXT
        )
    """)

    cursor.execute(
        "PRAGMA table_info(alert_history)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "fabricated_rate" not in columns:

        cursor.execute("""
            ALTER TABLE alert_history
            ADD COLUMN fabricated_rate REAL DEFAULT 0
        """)

    if "alert_text" not in columns:

        cursor.execute("""
            ALTER TABLE alert_history
            ADD COLUMN alert_text TEXT
        """)

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# SAVE ALERT HISTORY
# ============================================================

def save_history(
    severity,
    mitre_ids,
    fabricated_rate,
    alert_text
):

    connection = sqlite3.connect(
        DB_FILE
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO alert_history
        (
            timestamp,
            severity,
            mitre_ids,
            fabricated_rate,
            alert_text
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(
            timespec="seconds"
        ),
        severity,
        ", ".join(mitre_ids),
        fabricated_rate,
        alert_text
    ))

    connection.commit()
    connection.close()


# ============================================================
# INPUT SANITIZATION
# ============================================================

def sanitize_text(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        " ",
        text
    )

    text = text[:2000]

    return text.strip()


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(row):

    data = pd.DataFrame(
        [row]
    )

    data = data[FEATURES]

    data = pd.get_dummies(
        data,
        columns=CATEGORICAL_FEATURES,
        drop_first=False
    )

    for column in NUMERIC_FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [float("inf"), float("-inf")],
        0
    )

    data = data.fillna(0)

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


# ============================================================
# PREDICT SEVERITY
# ============================================================

def predict_severity(row):

    X = prepare_features(
        row
    )

    prediction = model.predict(
        X
    )[0]

    return str(
        prediction
    ).capitalize()


# ============================================================
# GET MODEL CATEGORIES
# ============================================================

def get_model_categories():

    proto_options = sorted(
        [
            column.replace(
                "proto_",
                "",
                1
            )
            for column in feature_columns
            if column.startswith(
                "proto_"
            )
        ]
    )

    service_options = sorted(
        [
            column.replace(
                "service_",
                "",
                1
            )
            for column in feature_columns
            if column.startswith(
                "service_"
            )
        ]
    )

    state_options = sorted(
        [
            column.replace(
                "state_",
                "",
                1
            )
            for column in feature_columns
            if column.startswith(
                "state_"
            )
        ]
    )

    # Safe fallbacks

    if not proto_options:

        proto_options = [
            "tcp",
            "udp",
            "icmp"
        ]

    if not service_options:

        service_options = [
            "http",
            "https",
            "ftp",
            "ssh",
            "dns",
            "smtp"
        ]

    if not state_options:

        state_options = [
            "FIN",
            "CON",
            "INT",
            "REQ",
            "RST"
        ]

    return (
        proto_options,
        service_options,
        state_options
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TriageMind SOC Copilot",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ TriageMind"
)

st.subheader(
    "RAG-Powered Security Operations Center Alert Triage Copilot"
)

st.caption(
    "Random Forest + Semantic Parsing + MITRE ATT&CK RAG + Ollama"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "TriageMind"
)

st.sidebar.info(
    """
Pipeline:

Network Telemetry
↓
Random Forest
↓
Semantic Parser
↓
MITRE ATT&CK RAG
↓
Ollama
↓
Citation Verifier
"""
)

st.sidebar.success(
    "System components loaded"
)


# ============================================================
# NETWORK ALERT INPUT
# ============================================================

st.header(
    "Network Alert Input"
)

col1, col2, col3 = st.columns(3)


# ============================================================
# GET DROPDOWN VALUES
# ============================================================

(
    proto_options,
    service_options,
    state_options
) = get_model_categories()


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    proto = st.selectbox(
        "Protocol",
        proto_options,
        index=(
            proto_options.index("tcp")
            if "tcp" in proto_options
            else 0
        )
    )

    service = st.selectbox(
        "Service",
        service_options,
        index=(
            service_options.index("http")
            if "http" in service_options
            else 0
        )
    )

    state = st.selectbox(
        "Connection State",
        state_options,
        index=(
            state_options.index("FIN")
            if "FIN" in state_options
            else 0
        )
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    dur = st.number_input(
        "Duration",
        min_value=0.0,
        value=2.5
    )

    spkts = st.number_input(
        "Source Packets",
        min_value=0.0,
        value=150.0
    )

    dpkts = st.number_input(
        "Destination Packets",
        min_value=0.0,
        value=120.0
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    sbytes = st.number_input(
        "Source Bytes",
        min_value=0.0,
        value=25000.0
    )

    dbytes = st.number_input(
        "Destination Bytes",
        min_value=0.0,
        value=18000.0
    )

    rate = st.number_input(
        "Traffic Rate",
        min_value=0.0,
        value=250.0
    )


# ============================================================
# ANALYST NOTES
# ============================================================

st.subheader(
    "Analyst Notes"
)

analyst_notes = st.text_area(
    "Describe the alert or add analyst observations",
    placeholder=(
        "Example: Multiple failed connections from the same "
        "source were observed within a short period."
    ),
    height=120,
    max_chars=2000
)


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

run_button = st.button(
    "🔍 Analyze Alert",
    type="primary",
    use_container_width=True
)


# ============================================================
# TRIAGE PIPELINE
# ============================================================

if run_button:

    # --------------------------------------------------------
    # SANITIZE ANALYST NOTES
    # --------------------------------------------------------

    analyst_notes = sanitize_text(
        analyst_notes
    )


    # --------------------------------------------------------
    # CREATE MODEL INPUT ROW
    # --------------------------------------------------------

    row = {
        "proto": sanitize_text(proto),
        "service": sanitize_text(service),
        "state": sanitize_text(state),
        "dur": dur,
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "rate": rate
    }


    # --------------------------------------------------------
    # 1. RANDOM FOREST
    # --------------------------------------------------------

    with st.spinner(
        "Classifying alert severity..."
    ):

        severity = predict_severity(
            row
        )


    # --------------------------------------------------------
    # 2. SEMANTIC PARSER
    # --------------------------------------------------------

    with st.spinner(
        "Generating semantic representation..."
    ):

        semantic_text = parse_alert(
            row
        )


    # --------------------------------------------------------
    # ADD ANALYST NOTES TO CONTEXT
    # --------------------------------------------------------

    if analyst_notes:

        semantic_text = (
            f"{semantic_text}\n\n"
            f"Analyst Notes:\n"
            f"{analyst_notes}"
        )


    # --------------------------------------------------------
    # 3. MITRE ATT&CK RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "Searching MITRE ATT&CK knowledge base..."
    ):

        retrieved = retrieve(
            semantic_text,
            top_k=3
        )


    # --------------------------------------------------------
    # CREATE RETRIEVED CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"{item['id']} - {item['name']}\n"
            f"{item['description']}"
        )

    retrieved_context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # 4. OLLAMA ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "Generating SOC analysis with Ollama..."
    ):

        # IMPORTANT:
        # Pass Random Forest predicted severity
        # to Ollama.

        llm_response = generate_analysis(
            semantic_text,
            retrieved_context,
            severity=severity
        )


    # --------------------------------------------------------
    # 5. CITATION VERIFICATION
    # --------------------------------------------------------

    verification = verify_citations(
        llm_response,
        retrieved_context
    )


    # --------------------------------------------------------
    # SAVE TO SQLITE
    # --------------------------------------------------------

    save_history(
        severity,
        verification["valid_ids"],
        verification["fabricated_id_rate"],
        semantic_text
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.success(
        "Alert analysis completed successfully."
    )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.header(
        "Triage Result"
    )

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Severity",
            severity
        )


    with result_col2:

        fabricated_rate = (
            verification["fabricated_id_rate"]
            * 100
        )

        st.metric(
            "Fabricated-ID Rate",
            f"{fabricated_rate:.2f}%"
        )


    # ========================================================
    # MITRE ATT&CK MAPPING
    # ========================================================

    st.header(
        "MITRE ATT&CK Mapping"
    )

    if retrieved:

        for item in retrieved:

            with st.expander(
                f"{item['id']} — {item['name']} "
                f"(Similarity: {item['similarity']:.4f})"
            ):

                st.write(
                    item["description"]
                )

    else:

        st.warning(
            "No MITRE ATT&CK techniques were retrieved."
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    st.header(
        "SOC Analyst Analysis"
    )

    st.write(
        llm_response
    )


    # ========================================================
    # CITATION VERIFICATION
    # ========================================================

    st.header(
        "Citation Verification"
    )

    verification_col1, verification_col2 = (
        st.columns(2)
    )


    with verification_col1:

        st.write(
            "**Cited IDs**"
        )

        st.write(
            verification["cited_ids"]
        )

        st.write(
            "**Valid IDs**"
        )

        st.write(
            verification["valid_ids"]
        )


    with verification_col2:

        st.write(
            "**Fabricated IDs**"
        )

        if verification["fabricated_ids"]:

            st.error(
                verification["fabricated_ids"]
            )

        else:

            st.success(
                "No fabricated technique IDs detected."
            )


    # ========================================================
    # VERIFICATION STATUS
    # ========================================================

    st.header(
        "Verification Status"
    )

    if fabricated_rate <= 5:

        st.success(
            f"Fabricated-ID rate is "
            f"{fabricated_rate:.2f}% "
            "— within the ≤5% target."
        )

    else:

        st.error(
            f"Fabricated-ID rate is "
            f"{fabricated_rate:.2f}% "
            "— above the ≤5% target."
        )


    # ========================================================
    # SEMANTIC REPRESENTATION
    # ========================================================

    with st.expander(
        "View Semantic Representation"
    ):

        st.write(
            semantic_text
        )


# ============================================================
# ALERT HISTORY
# ============================================================

st.divider()

st.header(
    "Alert History"
)


if os.path.exists(DB_FILE):

    connection = sqlite3.connect(
        DB_FILE
    )

    try:

        history = pd.read_sql_query(
            """
            SELECT
                timestamp,
                severity,
                mitre_ids,
                fabricated_rate
            FROM alert_history
            ORDER BY id DESC
            LIMIT 20
            """,
            connection
        )

    finally:

        connection.close()


    if len(history) > 0:

        # Convert rate to percentage
        history["fabricated_rate"] = (
            history["fabricated_rate"] * 100
        ).round(2)


        # Rename columns
        history = history.rename(
            columns={
                "timestamp": "Timestamp",
                "severity": "Severity",
                "mitre_ids": "MITRE Techniques",
                "fabricated_rate":
                    "Fabricated-ID Rate (%)"
            }
        )


        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No alert history yet."
        )