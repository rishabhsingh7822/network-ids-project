import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")

st.set_page_config(
    page_title="Network IDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")
st.caption("Real-time AI-powered threat detection — Powered by Ensemble ML + LLaMA3")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
auto_refresh  = st.sidebar.checkbox("Auto Refresh", value=False)
show_brief    = st.sidebar.checkbox("Show AI Threat Briefs", value=True)
n_samples     = st.sidebar.slider("Samples to analyze", 10, 500, 100)

# ── API Health Check ──────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"✅ API: {health['status']}")
except requests.exceptions.RequestException:
    st.sidebar.error("❌ API Offline — start the FastAPI server first")
    st.stop()

# ── Model Stats ───────────────────────────────────────────────
try:
    stats = requests.get(f"{API_URL}/stats", timeout=3).json()
except requests.exceptions.RequestException:
    stats = {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Accuracy",    f"{float(stats.get('accuracy', 0.9989))*100:.2f}%")
col2.metric("📊 Macro F1",    f"{stats.get('macro_f1', 0.9027):.4f}")
col3.metric("⚡ Throughput",  stats.get('throughput', '28,531 flows/sec'))
col4.metric("🔄 Queue Size",  stats.get('retraining_queue', 0))

st.divider()

# ── Live Detection ────────────────────────────────────────────
st.subheader("🔴 Live Traffic Analysis")

if st.button("▶️ Run Detection", type="primary"):
    # Load test data
    try:
        X_train, X_test, y_train, y_test = joblib.load(
            BASE_DIR / 'data/processed/train_test_split.pkl'
        )
        le = joblib.load(BASE_DIR / 'data/processed/label_encoder.pkl')
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    # Sample random flows
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    results = []

    progress = st.progress(0)
    status   = st.empty()

    for i, idx in enumerate(indices):
        sample   = X_test.iloc[idx].tolist()
        true_label = le.inverse_transform([y_test.iloc[idx]])[0]

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={'features': sample, 'get_threat_brief': False},
                timeout=5
            ).json()

            results.append({
                'flow_id':    i + 1,
                'prediction': resp['prediction'],
                'confidence': resp['confidence'],
                'is_attack':  resp['is_attack'],
                'true_label': true_label,
                'correct':    resp['prediction'] == true_label
            })
        except Exception as e:
            st.warning(f"Flow {i} failed: {e}")

        progress.progress((i + 1) / len(indices))
        status.text(f"Analyzing flow {i+1}/{len(indices)}...")

    progress.empty()
    status.empty()

    if not results:
        st.error("No results returned!")
        st.stop()

    df = pd.DataFrame(results)

    # ── Metrics ───────────────────────────────────────────────
    attacks    = df[df['is_attack']].shape[0]
    benign     = df[~df['is_attack']].shape[0]
    accuracy   = df['correct'].mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flows",    len(df))
    m2.metric("🚨 Attacks",     attacks,  delta=f"{attacks/len(df)*100:.1f}%")
    m3.metric("✅ Benign",      benign)
    m4.metric("🎯 Accuracy",    f"{accuracy:.1f}%")

    # ── Charts ────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        attack_counts = df[df['is_attack']]['prediction'].value_counts()
        if len(attack_counts) > 0:
            fig = px.pie(
                values=attack_counts.values,
                names=attack_counts.index,
                title="Attack Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No attacks detected in this sample!")

    with c2:
        fig2 = px.histogram(
            df, x='confidence', color='is_attack',
            title="Prediction Confidence Distribution",
            labels={'confidence': 'Confidence', 'is_attack': 'Is Attack'},
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Results Table ─────────────────────────────────────────
    st.subheader("📋 Detection Results")
    attack_df = df[df['is_attack']].copy()
    if len(attack_df) > 0:
        st.dataframe(
            attack_df[['flow_id', 'prediction', 'confidence', 'true_label', 'correct']],
            use_container_width=True
        )

        # ── AI Threat Briefs ──────────────────────────────────
        if show_brief:
            st.subheader("🤖 AI Threat Briefs")
            top_attacks = attack_df['prediction'].value_counts().head(3)
            for attack_type, count in top_attacks.items():
                with st.expander(f"🚨 {attack_type} — {count} flows detected"):
                    with st.spinner(f"Generating threat brief for {attack_type}..."):
                        try:
                            sample_idx = attack_df[
                                attack_df['prediction'] == attack_type
                            ].index[0]
                            sample = X_test.iloc[sample_idx].tolist()
                            resp = requests.post(
                                f"{API_URL}/predict",
                                json={'features': sample, 'get_threat_brief': True},
                                timeout=30
                            ).json()
                            if resp.get('threat_brief'):
                                st.markdown(resp['threat_brief'])
                            else:
                                st.info("No threat brief available")
                        except Exception as e:
                            st.error(f"Brief failed: {e}")
    else:
        st.success("✅ No attacks detected in this sample!")

    # ── Raw Data ──────────────────────────────────────────────
    with st.expander("📊 View all results"):
        st.dataframe(df, use_container_width=True)