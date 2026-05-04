import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Capacidad Portante", layout="wide", page_icon="🏗️")

# ─── ESTILOS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f4f6fa; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1a3a5c; font-size: 2rem; border-bottom: 3px solid #2e86de; padding-bottom: 8px; }
    h2, h3 { color: #1a3a5c; }
    .stButton>button {
        background: linear-gradient(135deg, #1a3a5c, #2e86de);
        color: white; border-radius: 8px; font-size: 1rem;
        padding: 0.6rem 2.5rem; font-weight: bold;
        border: none; width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #2e86de, #1a3a5c); }
    .resultado-card {
        background: #1a3a5c; color: white; border-radius: 10px;
        padding: 18px 24px; margin: 8px 0; text-align: center;
    }
    .resultado-card h4 { margin: 0 0 6px 0; font-size: 0.85rem; opacity: 0.8; }
    .resultado-card span { font-size: 2rem; font-weight: bold; color: #74b9ff; }
    .factor-box {
        background: #eaf4fb; border-left: 4px solid #2e86de;
        border-radius: 6px; padding: 10px 16px; margin: 4px 0; font-size: 0.9rem;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Análisis de Capacidad Portante de Cimentaciones")
st.markdown("**Método de Terzaghi & Ecuación General (Meyerhof/Brinch-Hansen)**")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")

    # --- Edificación ---
    st.subheader("🏢 Edificación")
    sotano = st.radio("¿Edificación con sótano?", ["NO", "SI"], horizontal=True)
    h = 0.0
    if sotano == "SI":
        h = st.number_input("Profundidad de sótano h (m)", value=3.0, step=0.5)

    # --- Geometría ---
    st.subheader("📐 Geometría de la Zapata")
    col1, col2 = st.columns(2)
    with col1:
        B_ini = st.number_input("B inicial (m)", value=0.8, step=0.2)
        Df_ini = st.number_input("Df inicial (m)", value=0.8, step=0.2)
    with col2:
        B_fin = st.number_input("B final (m)", value=3.0, step=0.2)
        Df_fin = st.number_input("Df final (m)", value=3.0, step=0.2)
    dB  = st.number_input("Incremento ΔB (m)", value=0.5, step=0.1)
    dDf = st.number_input("Incremento ΔDf (m)", value=0.5, step=0.1)

    # --- Tipo de zapata ---
    st.subheader("🔷 Tipo de Zapata")
    tipo = st.selectbox("Geometría", [
        "CUADRADA (L = B)", "RECTANGULAR (L = 2B)", "RECTANGULAR (L = 3B)",
        "RECTANGULAR (L = 5B)", "CORRIDA (L → ∞)"
    ])
    k_map = {"CUADRADA (L = B)": 1, "RECTANGULAR (L = 2B)": 2,
              "RECTANGULAR (L = 3B)": 3, "RECTANGULAR (L = 5B)": 5,
              "CORRIDA (L → ∞)": 100}
    k = k_map[tipo]

    # --- Carga inclinada ---
    st.subheader("📐 Condición de Carga")
    beta = st.selectbox("Ángulo de inclinación β (°)", [0, 5, 10, 15, 20, 25, 30])

    # --- Factor de seguridad ---
    st.subheader("🛡️ Factor de Seguridad")
    FS = st.number_input("FS", value=3.0, step=0.5)

    # --- Perfil estratigráfico ---
    st.subheader("🧱 Perfil Estratigráfico")
    if "df_estrato" not in st.session_state:
        st.session_state.df_estrato = pd.DataFrame({
            "Estrato": [1, 2],
            "Espesor (m)": [3.0, 5.0],
            "φ (°)": [30, 25],
            "c (t/m²)": [1.0, 2.0],
            "γ (t/m³)": [1.7, 1.8]
        })
    st.session_state.df_estrato = st.data_editor(
        st.session_state.df_estrato, num_rows="dynamic", use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════════════════

def get_estrato(df_e, Df):
    """Obtiene propiedades del estrato a la profundidad Df."""
    prof = 0.0
    for _, row in df_e.iterrows():
        prof += row["Espesor (m)"]
        if Df <= prof:
            return row
    return df_e.iloc[-1]

def factores_capacidad_terzaghi(phi_deg):
    """Factores Nq, Nc, Nγ — Terzaghi (1943)."""
    phi = np.radians(phi_deg)
    if phi_deg == 0:
        Nc, Nq, Ng = 5.7, 1.0, 0.0
    else:
        Nq = (np.e**(np.pi * np.tan(phi))) * (np.tan(np.radians(45) + phi/2)**2)
        Nc = (Nq - 1) / np.tan(phi)
        Ng = 0.5 * (Nq - 1) * np.tan(phi)
    return Nc, Nq, Ng

def factores_capacidad_meyerhof(phi_deg):
    """Factores Nq, Nc, Nγ — Meyerhof (1963)."""
    phi = np.radians(phi_deg)
    if phi_deg == 0:
        Nc, Nq, Ng = 5.14, 1.0, 0.0
    else:
        Nq = np.e**(np.pi * np.tan(phi)) * (np.tan(np.radians(45) + phi/2)**2)
        Nc = (Nq - 1) / np.tan(phi)
        Ng = (Nq - 1) * np.tan(1.4 * phi)
    return Nc, Nq, Ng

def calcular_q_sobrecarga(h, Df, gamma):
    return (h + Df) * gamma

def terzaghi(df_e, h, Df, B, FS, k):
    row = get_estrato(df_e, Df)
    c     = row["c (t/m²)"]
    gamma = row["γ (t/m³)"]
    phi   = row["φ (°)"]
    Nc, Nq, Ng = factores_capacidad_terzaghi(phi)
    q = calcular_q_sobrecarga(h, Df, gamma)

    # Factores de forma Terzaghi
    if k == 1:         # cuadrada
        Sc, Sg = 1.3, 0.8
    elif k == 100:     # corrida
        Sc, Sg = 1.0, 1.0
    else:              # rectangular
        L = k * B
        Sc = 1 + 0.2 * (B / L)
        Sg = 1 - 0.4 * (B / L)

    qult = Sc*c*Nc + q*Nq + Sg*0.5*gamma*B*Ng
    return round(qult / FS, 3), round(qult, 3)

def general_meyerhof(df_e, h, Df, B, FS, beta_deg, k):
    """Ecuación General — Meyerhof con factores de forma, profundidad e inclinación."""
    row = get_estrato(df_e, Df)
    c     = row["c (t/m²)"]
    gamma = row["γ (t/m³)"]
    phi   = row["φ (°)"]
    Nc, Nq, Ng = factores_capacidad_meyerhof(phi)
    q = calcular_q_sobrecarga(h, Df, gamma)

    phi_r = np.radians(phi)
    beta_r = np.radians(beta_deg)

    # Longitud efectiva
    L = max(k * B, B * 1.0001)  # evitar div/0
    if k == 100:
        L = 1000 * B

    # ── Factores de forma (Meyerhof) ────────────────────────────────────────
    if phi == 0:
        Sc = 1 + 0.2 * (B / L)
        Sq = 1.0
        Sg = 1.0
    else:
        Sc = 1 + 0.2 * (B / L) * np.tan(np.radians(45) + phi_r/2)**2
        Sq = 1 + 0.1 * (B / L) * np.tan(np.radians(45) + phi_r/2)**2
        Sg = Sq

    # ── Factores de profundidad (Meyerhof) ──────────────────────────────────
    ratio = Df / B
    if phi == 0:
        dc = 1 + 0.4 * ratio
        dq = 1.0
        dg = 1.0
    else:
        dc = 1 + 0.4 * ratio
        dq = 1 + 0.1 * ratio * np.tan(np.radians(45) + phi_r/2)**2
        dg = dq

    # ── Factores de inclinación (Meyerhof) ──────────────────────────────────
    if beta_deg == 0:
        ic = iq = ig = 1.0
    else:
        if phi == 0:
            ic = 1 - (beta_deg / 90)
            iq = 1.0
            ig = 1.0
        else:
            iq = (1 - beta_deg / phi)**2  if phi > beta_deg else 0
            ic = iq - (1 - iq) / (Nc * np.tan(phi_r)) if phi > 0 else 0
            ig = (1 - beta_r / phi_r)**2 if phi > beta_deg else 0

    factores = {
        "Nc": round(Nc, 3), "Nq": round(Nq, 3), "Nγ": round(Ng, 3),
        "Sc": round(Sc, 3), "Sq": round(Sq, 3), "Sγ": round(Sg, 3),
        "dc": round(dc, 3), "dq": round(dq, 3), "dγ": round(dg, 3),
        "ic": round(ic, 3), "iq": round(iq, 3), "iγ": round(ig, 3),
        "q (t/m²)": round(q, 3),
    }

    term_c   = c   * Nc * Sc * dc * ic
    term_q   = q   * Nq * Sq * dq * iq
    term_g   = 0.5 * gamma * B * Ng * Sg * dg * ig
    qult     = term_c + term_q + term_g

    return round(qult / FS, 3), round(qult, 3), factores, (round(term_c,3), round(term_q,3), round(term_g,3))

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN PRINCIPAL — ECUACIONES
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📐 Teoría & Ecuaciones", "📊 Resultados & Tablas", "📈 Gráficos"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Terzaghi (1943)")
        st.latex(r"q_{ult} = S_c \cdot c N_c + q N_q + S_\gamma \cdot 0.5 \gamma B N_\gamma")
        st.markdown("""
| Forma | Sc | Sγ |
|---|---|---|
| Cuadrada | 1.3 | 0.8 |
| Rectangular | 1+0.2B/L | 1−0.4B/L |
| Corrida | 1.0 | 1.0 |
""")
    with col_b:
        st.subheader("Ecuación General — Meyerhof (1963)")
        st.latex(r"""
q_{ult} = c N_c S_c d_c i_c + q N_q S_q d_q i_q + 0.5\,\gamma B N_\gamma S_\gamma d_\gamma i_\gamma
""")
        st.latex(r"""
N_q = e^{\pi\tan\phi}\tan^2\!\left(45+\tfrac{\phi}{2}\right), \quad
N_c = \frac{N_q-1}{\tan\phi}, \quad
N_\gamma = (N_q-1)\tan(1.4\phi)
""")

    st.divider()
    st.subheader("Factores de Forma, Profundidad e Inclinación (Meyerhof)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Forma**")
        st.latex(r"S_c = 1 + 0.2\frac{B}{L}\tan^2\!\left(45+\frac{\phi}{2}\right)")
        st.latex(r"S_q = S_\gamma = 1 + 0.1\frac{B}{L}\tan^2\!\left(45+\frac{\phi}{2}\right)")
    with col2:
        st.markdown("**Profundidad**")
        st.latex(r"d_c = 1 + 0.4\frac{D_f}{B}")
        st.latex(r"d_q = d_\gamma = 1 + 0.1\frac{D_f}{B}\tan^2\!\left(45+\frac{\phi}{2}\right)")
    with col3:
        st.markdown("**Inclinación**")
        st.latex(r"i_q = i_\gamma = \left(1-\frac{\beta}{\phi}\right)^2")
        st.latex(r"i_c = i_q - \frac{1-i_q}{N_c\tan\phi}")


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════
df_e = st.session_state.df_estrato.copy()

calcular = st.button("🧱 CALCULAR CAPACIDAD PORTANTE")

if calcular:
    resultados = []
    B_vals  = np.arange(B_ini,  B_fin  + 1e-6, dB)
    Df_vals = np.arange(Df_ini, Df_fin + 1e-6, dDf)

    ultimo_factores = {}

    for B in B_vals:
        for Df in Df_vals:
            qa_T, qult_T = terzaghi(df_e, h, Df, round(B,4), FS, k)
            qa_G, qult_G, fac, terms = general_meyerhof(df_e, h, Df, round(B,4), FS, beta, k)
            ultimo_factores = fac
            resultados.append({
                "B (m)": round(B, 2), "Df (m)": round(Df, 2),
                "qult Terzaghi (t/m²)": qult_T, "qa Terzaghi (t/m²)": qa_T,
                "qult General (t/m²)": qult_G,  "qa General (t/m²)": qa_G,
                "Δq (%)": round((qult_G - qult_T) / qult_T * 100, 1),
                "Término cN (t/m²)": terms[0],
                "Término qN (t/m²)": terms[1],
                "Término 0.5γBN (t/m²)": terms[2],
            })

    df_res = pd.DataFrame(resultados)
    st.session_state.df_res = df_res
    st.session_state.ultimo_factores = ultimo_factores
    st.session_state.B_vals  = B_vals
    st.session_state.Df_vals = Df_vals
    st.success("✅ Cálculo completado.")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTADOS & TABLAS
# ══════════════════════════════════════════════════════════════════════════════
if "df_res" in st.session_state:
    df_res = st.session_state.df_res
    fac    = st.session_state.ultimo_factores

    with tab2:
        # ── Factores del último caso ─────────────────────────────────────────
        st.subheader("🔢 Factores de Capacidad & Correctores (último caso)")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        grupos = [
            ("Capacidad de carga", ["Nc","Nq","Nγ"]),
            ("Forma",              ["Sc","Sq","Sγ"]),
            ("Profundidad",        ["dc","dq","dγ"]),
            ("Inclinación",        ["ic","iq","iγ"]),
        ]
        for col, (titulo, keys) in zip([col_f1,col_f2,col_f3,col_f4], grupos):
            with col:
                st.markdown(f"**{titulo}**")
                for k_name in keys:
                    val = fac.get(k_name, "—")
                    st.markdown(f'<div class="factor-box">{k_name} = <b>{val}</b></div>',
                                unsafe_allow_html=True)

        st.divider()

        # ── Tabla comparativa completa ────────────────────────────────────────
        st.subheader("📋 Tabla Comparativa Completa")
        cols_show = ["B (m)","Df (m)",
                     "qult Terzaghi (t/m²)","qa Terzaghi (t/m²)",
                     "qult General (t/m²)", "qa General (t/m²)", "Δq (%)"]
        st.dataframe(
            df_res[cols_show].style
              .format({c: "{:.2f}" for c in cols_show if "(" in c})
              .background_gradient(subset=["Δq (%)"], cmap="RdYlGn_r"),
            use_container_width=True, height=400
        )

        # ── Pivot qadm para B y Df ────────────────────────────────────────────
        st.subheader("📊 Capacidad Admisible — Tabla Pivote")
        metodo_pivot = st.selectbox("Método para tabla pivote",
                                    ["qa General (t/m²)", "qa Terzaghi (t/m²)"])
        pivot = df_res.pivot(index="Df (m)", columns="B (m)", values=metodo_pivot).round(2)
        pivot.columns = [f"B={c} m" for c in pivot.columns]
        pivot.index   = [f"Df={i} m" for i in pivot.index]
        st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)

        # ── Desglose de términos ──────────────────────────────────────────────
        st.subheader("🧩 Desglose de Términos — Ecuación General")
        cols_terms = ["B (m)","Df (m)",
                      "Término cN (t/m²)","Término qN (t/m²)","Término 0.5γBN (t/m²)",
                      "qult General (t/m²)"]
        st.dataframe(df_res[cols_terms].style.format("{:.3f}"), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # GRÁFICOS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📈 Configuración de Gráficos")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            graf_metodo = st.multiselect(
                "Métodos a graficar", ["qa General (t/m²)", "qa Terzaghi (t/m²)"],
                default=["qa General (t/m²)", "qa Terzaghi (t/m²)"]
            )
        with g_col2:
            eje_x = st.radio("Variable en eje X", ["Df (m)", "B (m)"], horizontal=True)

        colores_G = ["#2e86de","#0652dd","#1289A7","#006266","#12CBC4"]
        colores_T = ["#e55039","#c0392b","#e74c3c","#922b21","#f39c12"]

        B_vals  = st.session_state.B_vals
        Df_vals = st.session_state.Df_vals

        # ── GRÁFICO 1: qa vs Df/B por cada B/Df ─────────────────────────────
        st.markdown("---")
        st.markdown("#### 📉 Gráfico 1 — qa Admisible vs Variable")

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_facecolor("#f8f9fa")
        fig1.patch.set_facecolor("#ffffff")

        param_vals = B_vals if eje_x == "Df (m)" else Df_vals
        grupo_col  = "B (m)" if eje_x == "Df (m)" else "Df (m)"

        for i, pval in enumerate(param_vals):
            sub = df_res[np.isclose(df_res[grupo_col], pval, atol=0.001)]
            for met, cols in zip(["qa General (t/m²)","qa Terzaghi (t/m²)"],
                                  [colores_G, colores_T]):
                if met in graf_metodo:
                    label = f"{met.split()[1]} — {grupo_col.split()[0]}={pval:.1f}m"
                    ls    = "-" if "General" in met else "--"
                    ax1.plot(sub[eje_x], sub[met], marker="o", linestyle=ls,
                             color=cols[i % len(cols)], linewidth=2, label=label)

        ax1.set_xlabel(eje_x, fontsize=12)
        ax1.set_ylabel("q admisible (t/m²)", fontsize=12)
        ax1.set_title("Capacidad Portante Admisible", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=8, ncol=2, loc="best")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # ── GRÁFICO 2: Comparación métodos (scatter) ─────────────────────────
        st.markdown("---")
        st.markdown("#### 🔵 Gráfico 2 — Terzaghi vs General (Dispersión)")

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.set_facecolor("#f8f9fa")
        fig2.patch.set_facecolor("#ffffff")

        scatter = ax2.scatter(df_res["qa Terzaghi (t/m²)"], df_res["qa General (t/m²)"],
                              c=df_res["B (m)"], cmap="viridis", s=60, edgecolors="gray",
                              linewidths=0.5, alpha=0.85)
        mn = min(df_res["qa Terzaghi (t/m²)"].min(), df_res["qa General (t/m²)"].min())
        mx = max(df_res["qa Terzaghi (t/m²)"].max(), df_res["qa General (t/m²)"].max())
        ax2.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Igualdad")
        plt.colorbar(scatter, ax=ax2, label="B (m)")
        ax2.set_xlabel("qa Terzaghi (t/m²)", fontsize=12)
        ax2.set_ylabel("qa General / Meyerhof (t/m²)", fontsize=12)
        ax2.set_title("Comparación de Métodos", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        # ── GRÁFICO 3: Mapa de calor qa ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🌡️ Gráfico 3 — Mapa de Calor de qa")

        met_heat = st.selectbox("Método para mapa de calor",
                                ["qa General (t/m²)", "qa Terzaghi (t/m²)"], key="heat")
        pivot_heat = df_res.pivot(index="Df (m)", columns="B (m)", values=met_heat)

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        cmap_custom = LinearSegmentedColormap.from_list("azul",
                          ["#d6eaf8","#2e86de","#1a3a5c"], N=256)
        im = ax3.imshow(pivot_heat.values, aspect="auto", cmap=cmap_custom,
                        origin="lower")
        ax3.set_xticks(range(len(pivot_heat.columns)))
        ax3.set_xticklabels([f"{c:.1f}" for c in pivot_heat.columns])
        ax3.set_yticks(range(len(pivot_heat.index)))
        ax3.set_yticklabels([f"{i:.1f}" for i in pivot_heat.index])
        ax3.set_xlabel("B (m)", fontsize=12)
        ax3.set_ylabel("Df (m)", fontsize=12)
        ax3.set_title(f"Mapa de Calor — {met_heat}", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax3, label="qa (t/m²)")
        # anotaciones
        for i in range(len(pivot_heat.index)):
            for j in range(len(pivot_heat.columns)):
                val = pivot_heat.values[i, j]
                ax3.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=8, color="white" if val > pivot_heat.values.max()*0.6 else "#1a3a5c")
        st.pyplot(fig3)

        # ── GRÁFICO 4: Desglose de términos (barras apiladas) ─────────────────
        st.markdown("---")
        st.markdown("#### 📦 Gráfico 4 — Desglose por Términos (Barras Apiladas)")

        B_sel  = st.selectbox("B fija para desglose", sorted(df_res["B (m)"].unique()))
        sub4   = df_res[np.isclose(df_res["B (m)"], B_sel, atol=0.001)].copy()
        labels = [f"Df={r:.1f}" for r in sub4["Df (m)"]]

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        ax4.set_facecolor("#f8f9fa")
        fig4.patch.set_facecolor("#ffffff")
        x    = np.arange(len(sub4))
        w    = 0.45
        bot  = np.zeros(len(sub4))
        cols_t = ["#2e86de","#1abc9c","#e67e22"]
        labs_t = ["Término cNc", "Término qNq", "Término 0.5γBNγ"]
        for col, lab in zip(["Término cN (t/m²)","Término qN (t/m²)","Término 0.5γBN (t/m²)"],
                             labs_t):
            vals = sub4[col].values
            ax4.bar(x, vals, w, bottom=bot, label=lab,
                    color=cols_t[labs_t.index(lab)], edgecolor="white", linewidth=0.5)
            bot += vals

        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=30)
        ax4.set_ylabel("qult (t/m²)", fontsize=12)
        ax4.set_title(f"Desglose por Términos — B={B_sel:.1f} m", fontsize=14, fontweight="bold")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)
        st.pyplot(fig4)

        # ── GRÁFICO 5: qa vs B para Df fijo ──────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📐 Gráfico 5 — Efecto del Ancho B (Df Fija)")

        Df_sel = st.selectbox("Df fija", sorted(df_res["Df (m)"].unique()))
        sub5   = df_res[np.isclose(df_res["Df (m)"], Df_sel, atol=0.001)]

        fig5, ax5 = plt.subplots(figsize=(9, 4))
        ax5.set_facecolor("#f8f9fa")
        fig5.patch.set_facecolor("#ffffff")
        for met, color, ls in [("qa General (t/m²)","#2e86de","-"),
                                ("qa Terzaghi (t/m²)","#e55039","--")]:
            if met in graf_metodo:
                ax5.plot(sub5["B (m)"], sub5[met], marker="s", color=color,
                         linestyle=ls, linewidth=2.5, label=met)

        ax5.fill_between(sub5["B (m)"], sub5["qa Terzaghi (t/m²)"],
                         sub5["qa General (t/m²)"], alpha=0.12, color="#2e86de")
        ax5.set_xlabel("B (m)", fontsize=12)
        ax5.set_ylabel("qa admisible (t/m²)", fontsize=12)
        ax5.set_title(f"Efecto del Ancho de Zapata — Df={Df_sel:.1f} m", fontsize=14, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)

else:
    with tab2:
        st.info("⬆️ Configure los parámetros en el panel lateral y presione **CALCULAR**.")
    with tab3:
        st.info("⬆️ Configure los parámetros en el panel lateral y presione **CALCULAR**.")
