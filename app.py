"""
Presentación interactiva: ¿Cuánto crecieron las ventas de 2024 a 2025?
Ejecutar con:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="¿Cuánto crecimos?", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Paleta (validada para daltonismo) y estilos
# ---------------------------------------------------------------------------
AZUL, NARANJA, AQUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRIS = "#8a8984"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.idea {border-left: 5px solid #2a78d6; background: rgba(42,120,214,.08);
       padding: .9rem 1.1rem; border-radius: 6px; margin: .6rem 0 1rem 0;}
.alerta {border-left: 5px solid #eb6834; background: rgba(235,104,52,.08);
         padding: .9rem 1.1rem; border-radius: 6px; margin: .6rem 0 1rem 0;}
.paso {font-size: .85rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65;}
.grande {font-size: 3.2rem; font-weight: 700; line-height: 1.1;}
</style>
""", unsafe_allow_html=True)


def idea(texto):
    st.markdown(f'<div class="idea">💡 <b>Idea clave:</b> {texto}</div>', unsafe_allow_html=True)


def alerta(texto):
    st.markdown(f'<div class="alerta">⚠️ {texto}</div>', unsafe_allow_html=True)


def estilo(fig, alto=380, titulo=None):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50 if titulo else 20, b=10),
                      title=titulo, showlegend=fig.layout.showlegend,
                      legend=dict(orientation="h", y=1.12, x=0),
                      hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=True, zerolinecolor="rgba(128,128,128,.5)")
    return fig


def pct(x):
    return f"{x:+.1f} %"


# ---------------------------------------------------------------------------
# Datos y cálculos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar():
    df = pd.read_csv(Path(__file__).parent / "ventas_mensuales.csv", encoding="utf-8")
    df["periodo"] = pd.to_datetime(df["periodo"], format="%Y-%m")
    return df


@st.cache_data
def calcular(df):
    r = {}
    gr = lambda a, b: (b / a - 1) * 100

    v = df.groupby("anio")["ventas"].sum()
    u = df.groupby("anio")["cantidad"].sum()
    r["v"], r["u"] = v, u
    r["g_total"] = gr(*v)
    r["g_unid"] = gr(*u)
    r["g_precio"] = gr(*(v / u))

    sem = df.assign(sem=np.where(df["mes"] <= 6, "1er semestre", "2do semestre"))
    r["sem"] = sem.pivot_table(index="sem", columns="anio", values="ventas", aggfunc="sum")

    # Tipo de sucursal
    meses_s = df.groupby("sucursal")["periodo"].nunique()
    apertura = df.groupby("sucursal")["periodo"].min()
    tipo = {s: ("Estables" if meses_s[s] == 24 else
                "Abrieron en 2024" if apertura[s].year == 2024 else "Abrieron en 2025")
            for s in meses_s.index}
    dg = df.assign(grupo=df["sucursal"].map(tipo))
    c = dg.pivot_table(index="grupo", columns="anio", values="ventas", aggfunc="sum").fillna(0)
    c["n"] = dg.groupby("grupo")["sucursal"].nunique()
    c["pp"] = (c[2025] - c[2024]) / v[2024] * 100
    r["contrib"] = c.loc[["Estables", "Abrieron en 2024", "Abrieron en 2025"]]

    # Like-for-like
    est = df[df["sucursal"].isin(meses_s[meses_s == 24].index)]
    r["n_est"] = est["sucursal"].nunique()

    def g_de(d):
        x = d.groupby("anio")["ventas"].sum()
        return gr(*x)

    filas = {}
    for nombre, d in [("Todas las sucursales", df), ("Tiendas estables (LFL)", est),
                      ("LFL Region_A", est[est["region"] == "Region_A"]),
                      ("LFL Region_B", est[est["region"] == "Region_B"])]:
        filas[nombre] = {"Con P10": g_de(d), "Sin P10": g_de(d[d["producto"] != "P10"])}
    r["lfl"] = pd.DataFrame(filas).T

    p10 = df[df["producto"] == "P10"].groupby("periodo")["sucursal"].nunique()
    r["p10"] = p10.reindex(pd.date_range("2024-01-01", "2025-12-01", freq="MS"), fill_value=0)

    # Volumen / mezcla / descuento (tiendas estables)
    g = est.groupby(["producto", "anio"]).agg(u=("cantidad", "sum"), v=("ventas", "sum")).unstack("anio").fillna(0)
    u24, u25, v24, v25 = g["u"][2024], g["u"][2025], g["v"][2024], g["v"][2025]
    ef_vol = v24.sum() * (u25.sum() / u24.sum() - 1)
    a_precio24 = (u25 * (v24 / u24)).sum()
    r["efectos"] = pd.Series({
        "Volumen": ef_vol,
        "Mezcla": a_precio24 - v24.sum() - ef_vol,
        "Descuento": v25.sum() - a_precio24,
    }) / v24.sum() * 100
    r["unid_est"] = (u24.sum(), u25.sum())
    r["aporte_prod"] = ((v25 - v24) / v24.sum() * 100).sort_values()
    r["catalogo"] = df.groupby("producto")[["categoria", "precio"]].first()

    # Muchas medidas
    vm = df.groupby(["anio", "mes"])["ventas"].sum().unstack(0)
    yoy = (vm[2025] / vm[2024] - 1) * 100
    r["yoy"] = yoy
    ms = df.groupby(["anio", "periodo"]).agg(v=("ventas", "sum"), s=("sucursal", "nunique"))
    pares_sin = (df[df["producto"] != "P10"].groupby(["sucursal", "mes", "anio"])["ventas"]
                   .sum().unstack("anio").dropna())
    r["medidas"] = pd.Series({
        "Ventas totales ($)": r["g_total"],
        "Unidades totales": r["g_unid"],
        "Precio promedio por unidad": r["g_precio"],
        "Ventas totales sin P10": filas["Todas las sucursales"]["Sin P10"],
        "LFL (tiendas estables)": filas["Tiendas estables (LFL)"]["Con P10"],
        "LFL sin P10": filas["Tiendas estables (LFL)"]["Sin P10"],
        "Tienda-mes en ambos años, sin P10": gr(pares_sin[2024].sum(), pares_sin[2025].sum()),
        "Nº de sucursales activas": gr(*ms.groupby("anio")["s"].mean()),
        "Venta mensual por sucursal": gr(*(ms["v"] / ms["s"]).groupby("anio").mean()),
        "Promedio de crecimientos mensuales": yoy.mean(),
        "Mediana de crecimientos mensuales": yoy.median(),
        "Dic 2025 vs Dic 2024": yoy[12],
    })
    return r


df = cargar()
R = calcular(df)

# ---------------------------------------------------------------------------
# Navegación tipo presentación
# ---------------------------------------------------------------------------
DIAPOSITIVAS = [
    "Portada",
    "1 · Crecimiento total",
    "2 · ¿De qué tiendas viene?",
    "3 · A tiendas iguales (LFL)",
    "4 · Volumen, mezcla y descuento",
    "5 · ¿Y el 4 %? Unidades vs dinero",
    "6 · Muchas formas de medir",
    "Resumen y reto final",
]
if "slide" not in st.session_state:
    st.session_state.slide = 0

with st.sidebar:
    st.markdown("## 📈 ¿Cuánto crecimos?")
    st.caption("Ventas mensuales · 2024 → 2025")
    elegida = st.radio("Diapositivas", DIAPOSITIVAS, index=st.session_state.slide, label_visibility="collapsed")
    st.session_state.slide = DIAPOSITIVAS.index(elegida)
    st.divider()
    st.caption("Datos: `ventas_mensuales.csv` · "
               f"{len(df):,} filas · {df['sucursal'].nunique()} sucursales · {df['producto'].nunique()} productos")

s = st.session_state.slide
st.progress((s + 1) / len(DIAPOSITIVAS), text=f"Diapositiva {s + 1} de {len(DIAPOSITIVAS)}")


def navegacion():
    st.divider()
    c1, _, c3 = st.columns([1, 4, 1])
    if s > 0 and c1.button("⬅️ Anterior", width="stretch"):
        st.session_state.slide -= 1
        st.rerun()
    if s < len(DIAPOSITIVAS) - 1 and c3.button("Siguiente ➡️", width="stretch", type="primary"):
        st.session_state.slide += 1
        st.rerun()


v24, v25 = R["v"][2024], R["v"][2025]
u24, u25 = R["u"][2024], R["u"][2025]

# ---------------------------------------------------------------------------
# 0. Portada
# ---------------------------------------------------------------------------
if s == 0:
    st.markdown('<p class="paso">Análisis descriptivo · Aprendizaje de máquina</p>', unsafe_allow_html=True)
    st.title("¿Cuánto crecieron las ventas de 2024 a 2025?")
    st.markdown("#### Parece una pregunta con una sola respuesta… pero no lo es.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas en dinero", pct(R["g_total"]), help="Ventas totales 2025 vs 2024")
    c2.metric("Unidades vendidas", pct(R["g_unid"]), help="Cantidad total 2025 vs 2024")
    c3.metric("Tiendas iguales (LFL)", pct(R["medidas"]["LFL (tiendas estables)"]),
              help="Solo las sucursales abiertas los 24 meses")
    st.write("")
    st.markdown("""
Los tres números son **correctos**, pero responden **preguntas distintas**.
En esta presentación vamos a descubrir, paso a paso:

1. 📊 Cómo se calcula una **tasa de crecimiento**.
2. 🏪 Cuánto del crecimiento se debe a **abrir tiendas nuevas**.
3. 🔍 Qué es el crecimiento **like-for-like** (a tiendas iguales).
4. 🧩 Cómo separar el crecimiento en **volumen, mezcla y descuento**.
5. ⚖️ Por qué crecimos **4 % en unidades** pero **6 % en dinero**.
6. 📏 Cuántas formas distintas hay de medir "el crecimiento".
""")
    idea("antes de decir «crecimos X %», hay que definir <b>qué</b> se mide, "
         "<b>sobre qué base</b> y <b>cómo</b> se agrega.")

# ---------------------------------------------------------------------------
# 1. Crecimiento total
# ---------------------------------------------------------------------------
elif s == 1:
    st.markdown('<p class="paso">Paso 1</p>', unsafe_allow_html=True)
    st.header("El crecimiento total: la medida más simple")
    st.markdown("La **tasa de crecimiento** compara un periodo con otro:")
    st.latex(r"\text{Crecimiento} = \left(\frac{\text{Ventas}_{2025}}{\text{Ventas}_{2024}} - 1\right)\times 100")
    st.latex(rf"= \left(\frac{{{v25:,.0f}}}{{{v24:,.0f}}} - 1\right)\times 100 = \mathbf{{{R['g_total']:.1f}\,\%}}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=["2024", "2025"], y=[v24, v25], marker_color=[GRIS, AZUL],
                               text=[f"{v24/1e6:.2f} M", f"{v25/1e6:.2f} M"], textposition="outside",
                               hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        fig.update_yaxes(range=[0, v25 * 1.15], title="Ventas ($)")
        st.plotly_chart(estilo(fig, titulo="Ventas totales por año"), width="stretch")
    with c2:
        sem = R["sem"]
        fig = go.Figure()
        for anio, color in [(2024, GRIS), (2025, AZUL)]:
            fig.add_bar(name=str(anio), x=sem.index, y=sem[anio], marker_color=color,
                        hovertemplate=f"{anio} · %{{x}}: %{{y:,.0f}}<extra></extra>")
        for i, idx in enumerate(sem.index):
            g = (sem.loc[idx, 2025] / sem.loc[idx, 2024] - 1) * 100
            fig.add_annotation(x=idx, y=sem.loc[idx].max() * 1.08, text=f"<b>{pct(g)}</b>", showarrow=False)
        fig.update_layout(barmode="group", showlegend=True)
        fig.update_yaxes(range=[0, sem.values.max() * 1.18], title="Ventas ($)")
        st.plotly_chart(estilo(fig, titulo="¿Fue parejo durante el año?"), width="stretch")

    idea(f"la empresa vendió <b>{pct(R['g_total'])}</b> más, pero no de forma pareja: "
         "el crecimiento se concentró en el <b>primer semestre</b>.")
    alerta("Pregunta incómoda: ¿ese 6 % significa que <b>cada tienda</b> vende más? "
           "Durante el periodo <b>abrieron 25 sucursales nuevas</b>… 🤔")

    with st.expander("🐍 Ver el código"):
        st.code("""v = df.groupby("anio")["ventas"].sum()
crecimiento = (v[2025] / v[2024] - 1) * 100""")

# ---------------------------------------------------------------------------
# 2. Tipo de sucursal
# ---------------------------------------------------------------------------
elif s == 2:
    st.markdown('<p class="paso">Paso 2</p>', unsafe_allow_html=True)
    st.header("¿De qué tiendas viene el crecimiento?")
    st.markdown("""
Dividimos las sucursales en 3 grupos y medimos la **contribución** de cada uno, en **puntos porcentuales (p.p.)**:
""")
    st.latex(r"\text{Contribución}_g = \frac{\text{Ventas}_{g,2025} - \text{Ventas}_{g,2024}}{\text{Ventas totales}_{2024}}\times 100")
    st.caption("Al dividir entre el **total** de 2024, las contribuciones **suman exactamente** el crecimiento total.")

    c = R["contrib"]
    etiquetas = [f"{g}<br>({int(c.loc[g, 'n'])} tiendas)" for g in c.index]
    fig = go.Figure(go.Waterfall(
        x=etiquetas + ["<b>TOTAL</b>"], y=list(c["pp"]) + [0],
        measure=["relative"] * 3 + ["total"],
        text=[f"+{x:.1f} p.p." for x in c["pp"]] + [f"<b>{R['g_total']:.1f} %</b>"], textposition="outside",
        increasing=dict(marker=dict(color=AZUL)), totals=dict(marker=dict(color=VIOLETA)),
        connector=dict(line=dict(color=GRIS, dash="dot")),
        hovertemplate="%{x}: %{y:.2f} p.p.<extra></extra>"))
    fig.update_yaxes(title="Puntos porcentuales", range=[0, R["g_total"] * 1.2])
    st.plotly_chart(estilo(fig, 420, "Cascada del crecimiento 2024 → 2025"), width="stretch")

    nuevo = c.loc[["Abrieron en 2024", "Abrieron en 2025"], "pp"].sum()
    c1, c2 = st.columns(2)
    c1.metric("Por tiendas existentes", f"{c.loc['Estables', 'pp']:.1f} p.p.")
    c2.metric("Por tiendas nuevas (expansión)", f"{nuevo:.1f} p.p.",
              f"{nuevo / R['g_total'] * 100:.0f} % del crecimiento", delta_color="off")
    idea(f"<b>casi dos tercios</b> del crecimiento ({nuevo:.1f} de {R['g_total']:.1f} p.p.) vienen de "
         "<b>abrir tiendas</b>, no de que las tiendas existentes vendan más.")

    with st.expander("🤔 ¿Por qué las que abrieron en 2024 también aportan?"):
        st.write("Porque en 2024 vendieron solo algunos meses y en 2025 el año completo: "
                 "su aumento viene de **tener más meses abiertas**, no de vender más.")

# ---------------------------------------------------------------------------
# 3. Like-for-like
# ---------------------------------------------------------------------------
elif s == 3:
    st.markdown('<p class="paso">Paso 3</p>', unsafe_allow_html=True)
    st.header("Crecimiento a tiendas iguales: *like-for-like* (LFL)")
    st.markdown(f"""
En el comercio, la medida estándar del crecimiento **orgánico** es el **LFL**: se comparan **solo las tiendas
abiertas en ambos periodos completos** (aquí, {R['n_est']} sucursales). Así se elimina el efecto de las aperturas.
""")

    con_p10 = st.toggle("Incluir el producto P10", value=True,
                        help="P10 se dejó de vender a partir de julio de 2025")
    col = "Con P10" if con_p10 else "Sin P10"
    lfl = R["lfl"][col]
    colores = [NARANJA if x < 0 else AZUL for x in lfl]
    fig = go.Figure(go.Bar(y=lfl.index, x=lfl.values, orientation="h", marker_color=colores,
                           text=[pct(x) for x in lfl], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    fig.update_xaxes(range=[min(-6, lfl.min() - 2), max(9, lfl.max() + 2)], title="Crecimiento %")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilo(fig, 320, f"Crecimiento 2024 → 2025 ({col.lower()})"), width="stretch")

    if con_p10:
        idea(f"las mismas tiendas vendieron solo <b>{pct(lfl['Tiendas estables (LFL)'])}</b>. "
             f"Region_B creció <b>{pct(lfl['LFL Region_B'])}</b>, mientras que Region_A <b>cayó "
             f"{lfl['LFL Region_A']:.1f} %</b>. Prueba a quitar P10 con el interruptor 👆")
    else:
        idea(f"sin P10, el LFL sube a <b>{pct(lfl['Tiendas estables (LFL)'])}</b> y Region_A pasa a "
             f"<b>{pct(lfl['LFL Region_A'])}</b>. ¡Casi toda la caída de Region_A era P10!")

    st.subheader("¿Qué le pasó a P10?")
    p10 = R["p10"]
    fig = go.Figure(go.Scatter(x=p10.index, y=p10.values, mode="lines+markers", line=dict(color=NARANJA, width=2),
                               marker=dict(size=8), hovertemplate="%{x|%b %Y}: %{y} sucursales<extra></extra>"))
    fig.add_vrect(x0="2025-06-15", x1="2025-12-15", fillcolor=NARANJA, opacity=0.08, line_width=0)
    fig.add_annotation(x="2025-09-15", y=35, text="Retiro del producto", showarrow=False)
    fig.update_yaxes(title="Sucursales que lo venden", rangemode="tozero")
    st.plotly_chart(estilo(fig, 300), width="stretch")
    alerta("Un <b>evento puntual</b> (el retiro de un producto) puede cambiar mucho la lectura del crecimiento. "
           "Siempre hay que buscarlos.")

# ---------------------------------------------------------------------------
# 4. Volumen, mezcla, descuento
# ---------------------------------------------------------------------------
elif s == 4:
    st.markdown('<p class="paso">Paso 4</p>', unsafe_allow_html=True)
    st.header("¿Por qué crecieron las tiendas estables?")
    st.markdown("Recordemos que `ventas = cantidad × precio × (1 − descuento)`, y que **el precio de cada producto "
                "no cambió** entre años. Entonces las ventas solo pueden cambiar por tres razones:")
    c1, c2, c3 = st.columns(3)
    c1.markdown("##### 📦 Volumen\n¿Se vendieron **más unidades** en total?")
    c2.markdown("##### 🧩 Mezcla\n¿Se vendieron más productos **caros** y menos baratos?")
    c3.markdown("##### 🏷️ Descuento\n¿Se dio **más o menos** descuento?")

    ef = R["efectos"]
    fig = go.Figure(go.Waterfall(
        x=list(ef.index) + ["<b>Crecimiento LFL</b>"], y=list(ef.values) + [0],
        measure=["relative"] * 3 + ["total"],
        text=[f"{x:+.2f} p.p." for x in ef] + [f"<b>{ef.sum():.1f} %</b>"], textposition="outside",
        increasing=dict(marker=dict(color=AZUL)), decreasing=dict(marker=dict(color=NARANJA)),
        totals=dict(marker=dict(color=VIOLETA)), connector=dict(line=dict(color=GRIS, dash="dot")),
        hovertemplate="%{x}: %{y:.2f} p.p.<extra></extra>"))
    fig.update_yaxes(title="Puntos porcentuales", range=[-0.5, 3.6])
    st.plotly_chart(estilo(fig, 380, "Descomposición del crecimiento a tiendas iguales"), width="stretch")

    ua, ub = R["unid_est"]
    idea(f"las tiendas estables vendieron casi las <b>mismas unidades</b> ({ua:,.0f} → {ub:,.0f}, "
         f"{pct((ub / ua - 1) * 100)}). El crecimiento viene casi todo de la <b>mezcla</b>: "
         "se vendieron más productos caros.")

    st.subheader("¿Qué productos aportaron y cuáles restaron?")
    ap = R["aporte_prod"]
    cat = R["catalogo"]
    fig = go.Figure(go.Bar(
        x=ap.index, y=ap.values, marker_color=[NARANJA if x < 0 else AZUL for x in ap],
        customdata=np.stack([cat.loc[ap.index, "categoria"], cat.loc[ap.index, "precio"]], axis=1),
        hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>Precio: %{customdata[1]}<br>"
                      "Aporte: %{y:+.2f} p.p.<extra></extra>"))
    fig.update_yaxes(title="p.p. del crecimiento LFL")
    st.plotly_chart(estilo(fig, 330), width="stretch")
    st.caption("Pasa el cursor sobre cada barra para ver la categoría y el precio del producto. "
               "P05 (Alivio del dolor) es el gran motor; P08, P10 y P02 restaron.")

    with st.expander("🐍 Ver el cálculo"):
        st.code("""precio_neto_24 = v24 / u24                       # precio por unidad (ya con descuento) en 2024
ef_volumen   = v24.sum() * (u25.sum() / u24.sum() - 1)
ventas25_a_p24 = (u25 * precio_neto_24).sum()      # 2025 "valorado" a precios de 2024
ef_mezcla    = ventas25_a_p24 - v24.sum() - ef_volumen
ef_descuento = v25.sum() - ventas25_a_p24""")

# ---------------------------------------------------------------------------
# 5. Unidades vs dinero
# ---------------------------------------------------------------------------
elif s == 5:
    st.markdown('<p class="paso">Paso 5</p>', unsafe_allow_html=True)
    st.header("¿Y el 4 %? Crecer en unidades vs crecer en dinero")
    st.markdown("Las ventas en dinero se pueden escribir como **unidades × precio promedio por unidad**. "
                "Por eso los crecimientos **se multiplican**:")
    st.latex(r"(1 + g_{\text{ventas}}) = (1 + g_{\text{unidades}}) \times (1 + g_{\text{precio promedio}})")

    c1, c2, c3, c4 = st.columns([1, .3, 1, 1.3])
    c1.metric("📦 Unidades", pct(R["g_unid"]), f"{u24:,.0f} → {u25:,.0f}", delta_color="off")
    c2.markdown("<p class='grande' style='text-align:center'>×</p>", unsafe_allow_html=True)
    c3.metric("🏷️ Precio promedio", pct(R["g_precio"]),
              f"{v24 / u24:.2f} → {v25 / u25:.2f}", delta_color="off")
    c4.metric("💰 Ventas ($)", pct(R["g_total"]), "= resultado", delta_color="off")
    st.latex(rf"{1 + R['g_unid'] / 100:.3f} \times {1 + R['g_precio'] / 100:.3f} = "
             rf"{(1 + R['g_unid'] / 100) * (1 + R['g_precio'] / 100):.3f} \;\Rightarrow\; +{R['g_total']:.1f}\,\%")

    base = pd.DataFrame({"Unidades": R["u"] / R["u"][2024] * 100,
                         "Ventas ($)": R["v"] / R["v"][2024] * 100,
                         "Precio promedio": (R["v"] / R["u"]) / (v24 / u24) * 100})
    fig = go.Figure()
    for nombre, color in [("Ventas ($)", AZUL), ("Unidades", AQUA), ("Precio promedio", VIOLETA)]:
        fig.add_bar(name=nombre, x=["2024", "2025"], y=base[nombre], marker_color=color,
                    text=[f"{y:.1f}" for y in base[nombre]], textposition="outside",
                    hovertemplate=f"{nombre} · %{{x}}: %{{y:.1f}}<extra></extra>")
    fig.update_layout(barmode="group", showlegend=True)
    fig.update_yaxes(range=[90, 108], title="Índice (2024 = 100)")
    st.plotly_chart(estilo(fig, 360, "Las tres medidas con base 2024 = 100"), width="stretch")
    st.caption("Al poner las tres medidas en una **base común (2024 = 100)** se pueden comparar en un mismo eje, "
               "aunque tengan unidades distintas.")

    idea("<b>¡Aquí está el 4 %!</b> La empresa vendió <b>4 % más unidades</b>. Como además se vendieron "
         "productos más caros, el precio promedio subió ~1.9 % y las ventas en dinero crecieron <b>6 %</b>.")
    with st.expander("🤔 Si ningún precio cambió, ¿cómo sube el precio promedio?"):
        st.write("Por el **efecto mezcla**. Imagina que vendes 1 producto de 10 y 1 de 100: el promedio es 55. "
                 "Si al año siguiente vendes 1 de 10 y **2** de 100, el promedio sube a 70, ¡sin cambiar ningún precio!")

# ---------------------------------------------------------------------------
# 6. Muchas medidas
# ---------------------------------------------------------------------------
elif s == 6:
    st.markdown('<p class="paso">Paso 6</p>', unsafe_allow_html=True)
    st.header("Un mismo dato, muchas respuestas")
    st.markdown("Con los **mismos datos** calculamos el crecimiento de 12 formas distintas:")

    rango = st.slider("Resaltar las medidas que estén entre (%)", -5.0, 10.0, (3.5, 4.5), 0.5)
    m = R["medidas"].sort_values()
    dentro = m.between(*rango)
    fig = go.Figure(go.Bar(y=m.index, x=m.values, orientation="h",
                           marker_color=[AZUL if d else "rgba(138,137,132,.45)" for d in dentro],
                           text=[pct(x) for x in m], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    fig.add_vrect(x0=rango[0], x1=rango[1], fillcolor=AZUL, opacity=0.07, line_width=0)
    fig.update_xaxes(range=[m.min() - 3, m.max() + 3], title="Crecimiento 2024 → 2025 (%)")
    st.plotly_chart(estilo(fig, 470), width="stretch")
    st.caption(f"{dentro.sum()} de {len(m)} medidas caen en el rango elegido. ¡Mueve el deslizador!")

    st.subheader("Crecimiento de cada mes frente al mismo mes del año anterior")
    yoy = R["yoy"]
    fig = go.Figure(go.Bar(x=MESES, y=yoy.values, marker_color=[NARANJA if x < 0 else AZUL for x in yoy],
                           hovertemplate="%{x}: %{y:+.1f} %<extra></extra>"))
    fig.add_hline(y=R["g_total"], line_dash="dash", line_color=GRIS,
                  annotation_text=f"crecimiento anual {R['g_total']:.1f} %", annotation_position="top left")
    fig.update_yaxes(title="%")
    st.plotly_chart(estilo(fig, 320), width="stretch")

    idea("los crecimientos mes a mes son <b>muy volátiles</b> (enero −10 %, abril +23 %). Además, el "
         "<b>promedio de porcentajes no es el porcentaje del total</b>: el promedio de los 12 meses da "
         f"{yoy.mean():.1f} %, pero el anual es {R['g_total']:.1f} %.")
    alerta(f"Dato curioso: la <b>venta mensual por sucursal bajó "
           f"{R['medidas']['Venta mensual por sucursal']:.1f} %</b>. Hay más tiendas, pero las nuevas venden poco "
           "y bajan el promedio.")

# ---------------------------------------------------------------------------
# 7. Resumen y quiz
# ---------------------------------------------------------------------------
elif s == 7:
    st.header("Resumen: ¿cuánto crecimos?")
    resumen = pd.DataFrame({
        "Medida": ["💰 Total en dinero", "📦 Total en unidades", "🏪 LFL (tiendas iguales)",
                   "🏪 LFL sin P10", "🅱️ LFL Region_B", "🅰️ LFL Region_A"],
        "Crecimiento": [pct(R["g_total"]), pct(R["g_unid"]),
                        pct(R["lfl"].loc["Tiendas estables (LFL)", "Con P10"]),
                        pct(R["lfl"].loc["Tiendas estables (LFL)", "Sin P10"]),
                        pct(R["lfl"].loc["LFL Region_B", "Con P10"]),
                        pct(R["lfl"].loc["LFL Region_A", "Con P10"])],
        "¿Qué responde?": ["¿Cuánto más dinero vendió la empresa?", "¿Cuántas unidades más vendió?",
                           "¿Cuánto más venden las mismas tiendas?", "Igual, sin el producto retirado",
                           "Crecimiento orgánico de la región grande", "Región pequeña, afectada por P10"],
    })
    st.dataframe(resumen, hide_index=True, width="stretch")

    st.markdown("""
**La historia completa en una frase:**
> La empresa vende **6 % más en dinero** y **4 % más en unidades**, pero las tiendas existentes venden
> **casi las mismas unidades**: crece por **abrir tiendas** y por **vender productos más caros**.
""")
    idea("un solo número de «crecimiento» puede esconder historias muy distintas. "
         "Para un modelo de regresión, una simple tendencia temporal confundiría <b>expansión</b> con <b>demanda</b>: "
         "conviene incluir la <b>antigüedad de la sucursal</b>.")

    st.divider()
    st.subheader("🧠 Reto final")
    q1 = st.radio("1. Si un gerente dice «las tiendas venden 6 % más», ¿es correcto?",
                  ["Sí, las ventas crecieron 6 %",
                   "No: el 6 % incluye tiendas nuevas; a tiendas iguales fue ~2 %",
                   "No: las ventas cayeron"], index=None)
    if q1:
        if q1.startswith("No: el 6"):
            st.success("¡Correcto! Casi 2/3 del 6 % viene de la expansión. El crecimiento orgánico (LFL) es ~2.2 %.")
        else:
            st.error("Revisa el Paso 2: ¿de qué tiendas viene el crecimiento?")

    q2 = st.radio("2. Ningún producto cambió de precio, pero el precio promedio subió 1.9 %. ¿Por qué?",
                  ["Hubo inflación", "Se dio menos descuento",
                   "Se vendieron proporcionalmente más productos caros (efecto mezcla)"], index=None)
    if q2:
        if "mezcla" in q2:
            st.success("¡Exacto! Es el efecto mezcla (Paso 4 y 5).")
        else:
            st.error("Pista: los precios no cambiaron y el descuento incluso subió un poco. Revisa el Paso 4.")

    q3 = st.radio("3. ¿Por qué Region_A tiene un LFL negativo (−4.8 %)?",
                  ["Sus tiendas cerraron", "Principalmente por el retiro del producto P10",
                   "Porque tiene menos sucursales"], index=None)
    if q3:
        if "P10" in q3:
            st.success("¡Bien! Sin P10, Region_A crece +1.6 % (Paso 3).")
        else:
            st.error("Pista: prueba el interruptor de P10 en el Paso 3.")

    if q1 and q2 and q3 and q1.startswith("No: el 6") and "mezcla" in q2 and "P10" in q3:
        st.balloons()

navegacion()
