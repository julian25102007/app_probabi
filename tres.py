import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import google.generativeai as genai 

# ── Importa la configuración de Gemini desde el archivo separado ──
from api import consultar_gemini, construir_prompt

st.set_page_config(page_title="Prueba Z + Gemini 2.5", page_icon="📊", layout="wide")
st.title("📊 Prueba Z con Asistente Gemini 2.5 Flash")

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")
    temperatura = st.slider(
        "Temperatura del modelo", 0.0, 1.0, 0.7, 0.05,
        help="Mayor valor = respuestas más creativas."
    )
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
    datos = np.concatenate([
        rng.normal(mu_real - sigma_r, sigma_r * 0.5, n // 2),
        rng.normal(mu_real + sigma_r, sigma_r * 0.5, n - n // 2)
    ])

x_bar      = datos.mean()
skew       = stats.skew(datos)
kurt       = stats.kurtosis(datos)
sh_w, sh_p = stats.shapiro(datos[:5000])
q1, q3     = np.percentile(datos, [25, 75])
iqr        = q3 - q1
n_out      = int(((datos < q1 - 1.5 * iqr) | (datos > q3 + 1.5 * iqr)).sum())

# ── TABS ─────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📈 Distribución", "🔬 Prueba Z", "🤖 Asistente Gemini"])

# ═══ TAB 1: DISTRIBUCIÓN ═════════════════════════════════
with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n", n)
    c2.metric("Media", f"{x_bar:.3f}")
    c3.metric("Asimetría", f"{skew:.3f}")
    c4.metric("Outliers", n_out)

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
    st.info(
        f"**Shapiro-Wilk** W={sh_w:.4f}, p={sh_p:.5f} → "
        f"{'✅ No se rechaza normalidad' if normal_ok else '⚠️ Se rechaza normalidad'} | "
        f"Sesgo={skew:.3f} | Curtosis={kurt:.3f} | Outliers={n_out}"
    )

# ═══ TAB 2: PRUEBA Z ════════════════════════════════════
with t2:
    se     = sigma / np.sqrt(n)
    z_calc = (x_bar - mu0) / se

    if cola == "Bilateral":
        z_crit  = stats.norm.ppf(1 - alpha / 2)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_calc)))
        reject  = abs(z_calc) > z_crit
    elif cola == "Cola izquierda":
        z_crit  = -stats.norm.ppf(1 - alpha)
        p_value = stats.norm.cdf(z_calc)
        reject  = z_calc < z_crit
    else:
        z_crit  = stats.norm.ppf(1 - alpha)
        p_value = 1 - stats.norm.cdf(z_calc)
        reject  = z_calc > z_crit

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Z calculado", f"{z_calc:.4f}")
    m2.metric("Z crítico", f"±{z_crit:.4f}" if cola == "Bilateral" else f"{z_crit:.4f}")
    m3.metric("p-value", f"{p_value:.6f}")
    m4.metric("Decisión", "🔴 Rechazar H₀" if reject else "🟢 No rechazar H₀")

    if reject:
        st.error(f"Se **rechaza H₀** (μ={mu0}). Z={z_calc:.4f} supera el valor crítico. p={p_value:.6f} < α={alpha}")
    else:
        st.success(f"**No se rechaza H₀** (μ={mu0}). Z={z_calc:.4f} dentro de la región de no rechazo. p={p_value:.6f} ≥ α={alpha}")

    # Curva con zona de rechazo
    fig2, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(-4.5, 4.5, 800)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color="#7c3aed", lw=2.5)
    ax.fill_between(x, y, alpha=0.1, color="#7c3aed", label="No rechazo")

    if cola == "Bilateral":
        for mask in [x <= -z_crit, x >= z_crit]:
            ax.fill_between(x[mask], y[mask], color="#ef4444", alpha=0.55)
        ax.axvline(-z_crit, color="#ef4444", ls="--", lw=1.8)
        ax.axvline( z_crit, color="#ef4444", ls="--", lw=1.8, label=f"Z crit=±{z_crit:.3f}")
    elif cola == "Cola izquierda":
        ax.fill_between(x[x <= z_crit], y[x <= z_crit], color="#ef4444", alpha=0.55)
        ax.axvline(z_crit, color="#ef4444", ls="--", lw=1.8, label=f"Z crit={z_crit:.3f}")
    else:
        ax.fill_between(x[x >= z_crit], y[x >= z_crit], color="#ef4444", alpha=0.55)
        ax.axvline(z_crit, color="#ef4444", ls="--", lw=1.8, label=f"Z crit={z_crit:.3f}")

    ax.axvline(np.clip(z_calc, -4.4, 4.4), color="#22c55e", lw=2.5, label=f"Z calc={z_calc:.3f}")
    ax.fill_between([], [], color="#ef4444", alpha=0.55, label="Zona de rechazo")
    ax.set_title(f"Distribución Normal — {cola} | α={alpha}", fontsize=12)
    ax.legend(fontsize=9); ax.set_xlabel("Z"); ax.set_ylabel("f(Z)")
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    st.session_state["zr"] = dict(
        x_bar=x_bar, mu0=mu0, n=n, sigma=sigma, se=se,
        z_calc=z_calc, z_crit=z_crit, p_value=p_value, alpha=alpha,
        cola=cola, reject=reject, skew=skew, kurt=kurt, sh_p=sh_p, n_out=n_out
    )

# ═══ TAB 3: ASISTENTE GEMINI ════════════════════════════
with t3:
    if "zr" not in st.session_state:
        st.info("Primero ejecuta la Prueba Z en la pestaña anterior.")
    else:
        zr = st.session_state["zr"]

        st.caption("🤖 Powered by **Gemini 2.5 Flash** · Google AI")

        pregunta = st.text_area(
            "Pregunta adicional (opcional)",
            placeholder="¿Qué pasaría si n fuera mayor? ¿Cómo afecta el α la decisión?"
        )

        if st.button("🚀 Consultar a Gemini"):
            prompt_completo = construir_prompt(zr, pregunta)

            with st.spinner("Consultando a Gemini 2.5 Flash..."):
                respuesta = consultar_gemini(prompt_completo, temperatura)

            st.markdown("### 🧠 Respuesta de Gemini")
            st.markdown(respuesta)
            st.divider()
            st.info(
                f"**Decisión automática:** {'🔴 Rechazar H₀' if zr['reject'] else '🟢 No rechazar H₀'} "
                f"| Z={zr['z_calc']:.4f} | p={zr['p_value']:.6f}"
            )

            reporte = (
                "REPORTE — PRUEBA Z\n"
                "==================\n\n"
                f"{prompt_completo}\n\n"
                "RESPUESTA DE GEMINI 2.5 FLASH\n"
                "==============================\n\n"
                f"{respuesta}"
            )
            st.download_button(
                "⬇️ Descargar reporte",
                data=reporte,
                file_name="reporte_z_gemini.txt",
                mime="text/plain"
            )