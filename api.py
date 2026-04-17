# api.py

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── CARGAR .env ──────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ No se encontró la API KEY de Gemini")

# ── CONFIG ───────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

_cliente = None

def _get_cliente():
    global _cliente
    if _cliente is None:
        _cliente = genai.Client(api_key=GEMINI_API_KEY)
    return _cliente


# ── CONSULTA A GEMINI ─────────────────────────
def consultar_gemini(prompt: str, temperatura: float = 0.7) -> str:
    try:
        cliente = _get_cliente()

        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Eres un asistente experto en estadística, "
                    "explica de forma clara, didáctica y en español."
                ),
                temperature=temperatura,
                max_output_tokens=2048,
            ),
        )

        return respuesta.text if respuesta.text else "⚠️ Sin respuesta."

    except Exception as e:
        return f"❌ Error Gemini: {e}"


# ── CONSTRUIR PROMPT ─────────────────────────
def construir_prompt(zr: dict, pregunta_extra: str = "") -> str:

    base = f"""Prueba Z de una muestra:
- x̄={zr['x_bar']:.4f}, μ₀={zr['mu0']}, n={zr['n']}
- σ={zr['sigma']:.4f}, SE={zr['se']:.4f}
- Z calculado={zr['z_calc']:.4f}, Z crítico={zr['z_crit']:.4f}
- p-value={zr['p_value']:.6f}
- α={zr['alpha']} | prueba={zr['cola']}
- decisión={'RECHAZAR H₀' if zr['reject'] else 'NO rechazar H₀'}

Diagnóstico:
- Shapiro p={zr['sh_p']:.5f}
- Asimetría={zr['skew']:.3f}
- Curtosis={zr['kurt']:.3f}
- Outliers={zr['n_out']}

Explica:
1. Interpretación
2. Supuestos
3. Conclusión clara"""

    if pregunta_extra.strip():
        base += f"\n\nPregunta: {pregunta_extra}"

    return base