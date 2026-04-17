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

    # ── GENERACIÓN DE DATOS ───────────────────────────────────
rng = np.random.default_rng(int(seed))
if dist == "Normal":
    datos = rng.normal(mu_real, sigma_r, n)
elif dist == "Sesgada positiva":
    datos = rng.lognormal(np.log(max(mu_real, 1)), 0.5, n)
else:
    datos = np.concatenate([rng.normal(mu_real - sigma_r, sigma_r * 0.5, n // 2),
                            rng.normal(mu_real + sigma_r, sigma_r * 0.5, n - n // 2)])

x_bar     = datos.mean()
std_datos = datos.std()
skew      = stats.skew(datos)
kurt      = stats.kurtosis(datos)
sh_w, sh_p = stats.shapiro(datos[:5000])
q1, q3    = np.percentile(datos, [25, 75])
iqr       = q3 - q1
n_out     = int(((datos < q1 - 1.5 * iqr) | (datos > q3 + 1.5 * iqr)).sum())

# ── TABS ─────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📈 Distribución", "🔬 Prueba Z", "🤖 Asistente IA"])

# ═══ TAB 1: DISTRIBUCIÓN ═════════════════════════════════
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n",         n)
    c2.metric("Media",     f"{x_bar:.3f}")
    c3.metric("Asimetría", f"{skew:.3f}")
    c4.metric("Outliers",  n_out)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#f8f9fa")

    # Histograma + KDE
    axes[0].hist(datos, bins="auto", color="#7c3aed", alpha=0.6, edgecolor="white", density=True)
    xk = np.linspace(datos.min(), datos.max(), 300)
    axes[0].plot(xk, stats.gaussian_kde(datos)(xk), color="#0ea5e9", lw=2.5, label="KDE")
    axes[0].axvline(x_bar, color="#22c55e", lw=2, ls="--", label=f"Media={x_bar:.2f}")
    axes[0].set_title("Histograma + KDE"); axes[0].legend(fontsize=8)

    # Boxplot
    axes[1].boxplot(datos, patch_artist=True,
        boxprops=dict(facecolor="#ede9fe", color="#7c3aed"),
        medianprops=dict(color="#22c55e", lw=2.5),
        flierprops=dict(marker="o", color="#ef4444", alpha=0.6))
    axes[1].set_title("Boxplot")

    # Q-Q
    (osm, osr), (slope, intercept, r) = stats.probplot(datos, dist="norm")
    axes[2].scatter(osm, osr, color="#7c3aed", alpha=0.5, s=15)
    xl = np.array([min(osm), max(osm)])
    axes[2].plot(xl, slope * xl + intercept, color="#0ea5e9", lw=2, label=f"R²={r**2:.4f}")
    axes[2].set_title("Q-Q Plot"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    normal_ok = sh_p > 0.05
    st.info(f"**Shapiro-Wilk** W={sh_w:.4f}, p={sh_p:.5f} → "
            f"{'✅ No se rechaza normalidad' if normal_ok else '⚠️ Se rechaza normalidad'} | "
            f"Sesgo={skew:.3f} | Curtosis={kurt:.3f} | Outliers={n_out}")
