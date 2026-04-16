import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import google.generativeai as genai

st.set_page_config(page_title="Prueba Z + IA", page_icon="📊", layout="wide")
st.title("📊 Prueba Z con Asistente IA")

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    gemini_key = st.text_input("API Key de Gemini", type="password", placeholder="AIza...")
    st.divider()
    st.subheader("Datos sintéticos")
    n       = st.slider("n (muestra)", 30, 1000, 100)
    mu_real = st.number_input("Media real (μ real)", value=52.0)
    sigma_r = st.number_input("σ real", value=10.0, min_value=0.1)
    seed    = st.number_input("Semilla", value=42)
    dist    = st.selectbox("Distribución", ["Normal", "Sesgada positiva", "Bimodal"])
    st.divider()
    st.subheader("Prueba de hipótesis")
    mu0   = st.number_input("H₀: μ₀", value=50.0)
    sigma = st.number_input("σ poblacional conocida", value=10.0, min_value=0.01)
    alpha = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05)
    cola  = st.selectbox("Tipo de cola", ["Bilateral", "Cola izquierda", "Cola derecha"])