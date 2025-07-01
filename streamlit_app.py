import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import math

# constantes
gamma_w = 9.81 # kN/m3

# funciones
def sanity_check(df, verbose=True):
    # profundidad mayor a 0
    problematic = df[df["depth"] <= 0]
    if not problematic.empty:
        st.warning("Se encontraron profundidades menores o iguales a 0. Filtrando...")
        df = df[df["depth"] > 0]
        
    # profundidad estrictamente creciente
    if not df["depth"].is_monotonic_increasing:
        st.warning("Profundidades no monotónicas. Reordenando por profundidad...")
        df = df.sort_values("depth").reset_index(drop=True)
    
    # sin duplicados
    df = df.drop_duplicates(subset="depth")
    
    # qc y fs mayores a 0
    problematic = df[(df["qc"] <= 0) | (df["fs"] <= 0)]
    if not problematic.empty:
        st.warning("Valores de qc o fs menores o iguales a 0 encontrados. Filtrando...")
        df = df[(df["qc"] > 0) & (df["fs"] > 0)]
    
    return df

def ccf_values(series1, series2):
    p = (series1 - np.mean(series1)) / (np.std(series1) * len(series1))
    q = (series2 - np.mean(series2)) / (np.std(series2))  
    c = np.correlate(p, q, 'full')
    return c

def plot_ccf(lags, ccf, max_lag):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=ccf, mode='lines', name='CCF'))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.add_vline(x=max_lag, line_dash="dot", line_color="red")
    fig.update_layout(
        title="Cross-correlation qc vs fs",
        xaxis_title="Lags",
        yaxis_title="Correlation coefficient",
        height=400
    )
    return fig

def calculate_Ic_SBTn(df_in, delta_n): 
    while (delta_n > 0.01).any():
        df_in["CN"] = (pa / df_in["s'vo (kPa)"]) ** df_in["n"]
        df_in["Qtn"] = ((df_in["qt (MPa)"] * 1000 - df_in["svo (kPa)"]) / pa) * df_in["CN"]
        df_in["Ic SBTn"] = ((3.47 - np.log10(df_in["Qtn"]))**2 + (np.log10(df_in["Fr (%)"]) + 1.22)**2)**0.5
        n_prev = df_in["n"]
        df_in["n"] = 0.381 * df_in["Ic SBTn"] + 0.05 * (df_in["s'vo (kPa)"] / pa) - 0.15
        delta_n = abs(df_in["n"] - n_prev)

    mask = df_in["n"] > 1
    if mask.any():
        df_in.loc[mask, "n"] = 1
        df_in.loc[mask, "CN"] = (pa / df_in.loc[mask, "s'vo (kPa)"]) ** df_in.loc[mask, "n"]
        df_in.loc[mask, "Qtn"] = ((df_in.loc[mask, "qt (MPa)"] * 1000 - df_in.loc[mask, "svo (kPa)"]) / pa) * df_in.loc[mask, "CN"]
        df_in.loc[mask, "Ic SBTn"] = ((3.47 - np.log10(df_in.loc[mask, "Qtn"]))**2 + (np.log10(df_in.loc[mask, "Fr (%)"]) + 1.22)**2)**0.5

    ## print(max(delta_n))
    return df_in

#### CÓDIGO #####

st.set_page_config(page_title="Procesamiento de CPTu", layout="wide")

st.title("Procesador de ensayos CPTu")

st.markdown(
    """
    Esta aplicación permite subir un archivo de ensayo CPTu, realizar análisis de consistencia, 
    graficar y exportar resultados siguiendo las metodologías de Robertson, Boulanger, Idriss, etc.
    """
)

# --- 1. Subir archivo ---
uploaded_file = st.file_uploader(
    "Sube el archivo Excel con los datos CPTu:",
    type=["xlsx", "xls"]
)

if uploaded_file:
    # leer el archivo
    try:
        df = pd.read_excel(uploaded_file)
        st.success("Archivo cargado correctamente:")
        st.write(df.head())
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()
    
    # --- 2. Parámetros de entrada ---
    st.subheader("Parámetros generales")

    gamma = st.number_input("Peso unitario del material γ (kN/m³)", value=19.0)
    elev_known = st.checkbox("¿Se conoce la elevación del terreno?", value=False)
    elev = None
    if elev_known:
        elev = st.number_input("Elevación (m.s.n.m)", value=0.0)
    else:
        elev = None
    
    wt_known = st.checkbox("¿Se conoce el nivel freático?", value=False)
    wt = None
    if wt_known:
        wt = st.number_input("Nivel freático (m)", value=1.0)
    else:
        wt = None

    # parámetros avanzados
    st.subheader("Parámetros avanzados")

    an = st.number_input(
        "Área neta del cono (-)", 
        value=0.75,
        min_value=0.01,
        step=0.01,
        help="Relación de área neta del cono (valor típico entre 0.75 y 0.85)"
    )

    pa = st.number_input(
        "Presión atmosférica (kPa)", 
        value=101.325,
        min_value=1.000,
        step=0.001,
        help="Presión atmosférica de referencia"
    )

    Nkt = st.number_input(
        "Factor Nkt (para resistencia no drenada)", 
        value=14.0,
        min_value=1.0,
        step=0.1,
        help="Factor empleado para el cálculo de la resistencia no drenada"
    )
    
    # Conversión de unidades
    st.subheader("Conversión de unidades")

    qc_in_mpa = st.radio("¿La resistencia de punta qc está en MPa?", options=["Sí", "No"], key="qc_radio")
    fs_in_kpa = st.radio("¿La fricción fs está en kPa?", options=["Sí", "No"], key="fs_radio")
    u2_in_kpa = st.radio("¿La presión u2 está en kPa?", options=["Sí", "No"], key="u2_radio")

    factor_qc = 1.0
    factor_fs = 1.0
    factor_u2 = 1.0

    if qc_in_mpa == "No":
        factor_qc = st.number_input("Factor de conversión para qc:", value=1.0, key="factor_qc")
    if fs_in_kpa == "No":
        factor_fs = st.number_input("Factor de conversión para fs:", value=1.0, key="factor_fs")
    if u2_in_kpa == "No":
        factor_u2 = st.number_input("Factor de conversión para u2:", value=1.0, key="factor_u2")


    if st.button("Procesar archivo"):
        st.subheader("Procesando archivo...")

        # aplicar conversión de unidades
        df["qc"] = df["qc"] * factor_qc
        df["fs"] = df["fs"] * factor_fs
        df["u2"] = df["u2"] * factor_u2

        # ejecutar el chequeo de consistencia
        df_checked = sanity_check(df)

        st.success("Sanity check aplicado correctamente. Muestra del resultado:")
        st.dataframe(df_checked.head())

        # locally smooth
        window_pts = 2
        reject_nstd = 1.2
    
        for idx in df_checked.index[window_pts:-window_pts]:
            region = df_checked.loc[idx - window_pts:idx + window_pts][['qc', 'fs']].values
            region = np.delete(region, window_pts, axis=0)
            qc_mean, fs_mean = np.mean(region, axis=0)
            qc_std, fs_std = np.std(region, axis=0, ddof=1)
            if qc_std == 0 or fs_std == 0:
                continue
            qc_z = abs(df_checked.at[idx, 'qc'] - qc_mean) / qc_std
            fs_z = abs(df_checked.at[idx, 'fs'] - fs_mean) / fs_std
            if qc_z > reject_nstd or fs_z > reject_nstd:
                df_checked.at[idx, 'qc'] = qc_mean
                df_checked.at[idx, 'fs'] = fs_mean
    
        st.success("Se aplicó suavizado local.")

        st.session_state.df = df_checked
        st.success("Archivo procesado y almacenado en sesión.")

    if "df" in st.session_state:
        df_working = st.session_state.df.copy()
    
        # cross-correlation
        lags = signal.correlation_lags(len(df_working["fs"]), len(df_working["qc"]))
        ccf = ccf_values(df_working["fs"], df_working["qc"])
        valid_idx = np.where((lags >= -20) & (lags <= 20))[0]
        valid_lags = lags[valid_idx]
        valid_ccf = ccf[valid_idx]
        max_corr_idx = np.argmax(valid_ccf)
        max_lag = valid_lags[max_corr_idx]
    
        st.plotly_chart(plot_ccf(valid_lags, valid_ccf, max_lag))
    
        shift_ok = st.checkbox(f"¿Aplicar shift con lag {max_lag}?", value=False)
        if st.button("Aplicar shift"):
            if shift_ok and max_lag != 0:
                if max_lag < 0:
                    df_working = df_working.iloc[abs(max_lag):].reset_index(drop=True)
                else:
                    df_working["fs"] = df_working["fs"].shift(-max_lag)
                    df_working = df_working.dropna().reset_index(drop=True)
                st.success("Shift aplicado.")
            else:
                st.info("No se aplicó shift.")
            st.session_state.df = df_working
    
        # plot qc, fs, u2 con plotly
        if "df" in st.session_state:
            df_plot = st.session_state.df.copy()
        
            fig = make_subplots(
                rows=1, cols=3,
                shared_yaxes=True,
                horizontal_spacing=0.05,
                subplot_titles=["qc (MPa)", "fs (kPa)", "u2 (kPa)"]
            )
            fig.add_trace(
                go.Scatter(x=df_plot["qc"], y=df_plot["depth"], mode="lines", line=dict(color="blue")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_plot["fs"], y=df_plot["depth"], mode="lines", line=dict(color="red")),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df_plot["u2"], y=df_plot["depth"], mode="lines", line=dict(color="green")),
                row=1, col=3
            )
            fig.update_yaxes(autorange="reversed", title="Profundidad (m)", row=1, col=1)
            fig.update_layout(height=700, width=1200, title="Perfiles qc, fs y u2", showlegend=False)
            st.plotly_chart(fig)

            df_working = st.session_state.df.copy()
    
            # cálculo de elevación
            if elev is not None and not math.isnan(elev):
                df_working["elevation (m.s.n.m.)"] = elev - df_working["depth"]
            else:
                df_working["elevation (m.s.n.m.)"] = np.nan
            
            # nivel freático
            if wt is not None and not math.isnan(wt):
                h = df_working["depth"] - wt
                h[h < 0] = 0
                df_working["u0 (kPa)"] = gamma_w * h
            else:
                df_working["u0 (kPa)"] = 0
            
            # qt
            df_working["qt (MPa)"] = df_working["qc"] + (df_working["u2"] / 1000) * (1 - an)
            
            # sigma vo
            df_working["svo (kPa)"] = gamma * df_working["depth"]
            df_working["s'vo (kPa)"] = df_working["svo (kPa)"] - df_working["u0 (kPa)"]
            
            # Qt1
            df_working["Qt1"] = (df_working["qt (MPa)"] * 1000 - df_working["svo (kPa)"]) / df_working["s'vo (kPa)"]
            
            # Fr
            df_working["Fr (%)"] = df_working["fs"] / (df_working["qt (MPa)"] * 1000 - df_working["svo (kPa)"]) * 100
            
            # Ic inicial
            df_working["Ic SBTn"] = ((3.47 - np.log10(df_working["Qt1"]))**2 + (np.log10(df_working["Fr (%)"]) + 1.22)**2)**0.5
            
            # n inicial
            df_working["n"] = 0.381 * df_working["Ic SBTn"] + 0.05 * (df_working["s'vo (kPa)"] / pa) - 0.15
            
            # delta n
            delta_n = 100 - df_working["n"]
            
            # iterar
            df_working = calculate_Ic_SBTn(df_working, delta_n)
            
            # actualizar session_state
            st.session_state.df = df_working

            # Graficar
            fig = make_subplots(
                rows=1, cols=5,
                shared_yaxes=True,
                horizontal_spacing=0.05,
                subplot_titles=[
                    "qc / qt (MPa)", 
                    "σvo / σ'vo / u0 (kPa)", 
                    "Fr (%)", 
                    "Qt1", 
                    "Ic SBTn"
                ]
            )
            
            # Columna 1: qc y qt
            fig.add_trace(
                go.Scatter(
                    x=df_working["qc"],
                    y=df_working["depth"],
                    mode="lines",
                    name="qc",
                    line=dict(color="blue")
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_working["qt (MPa)"],
                    y=df_working["depth"],
                    mode="lines",
                    name="qt",
                    line=dict(color="orange")
                ),
                row=1, col=1
            )
            
            # Columna 2: σvo, σ'vo, u0
            fig.add_trace(
                go.Scatter(
                    x=df_working["svo (kPa)"],
                    y=df_working["depth"],
                    mode="lines",
                    name="σvo",
                    line=dict(color="green")
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=df_working["s'vo (kPa)"],
                    y=df_working["depth"],
                    mode="lines",
                    name="σ'vo",
                    line=dict(color="darkgreen")
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=df_working["u0 (kPa)"],
                    y=df_working["depth"],
                    mode="lines",
                    name="u0",
                    line=dict(color="lightblue")
                ),
                row=1, col=2
            )
            
            # Columna 3: Fr
            fig.add_trace(
                go.Scatter(
                    x=df_working["Fr (%)"],
                    y=df_working["depth"],
                    mode="lines",
                    name="Fr",
                    line=dict(color="red")
                ),
                row=1, col=3
            )
            
            # Columna 4: Qt1
            fig.add_trace(
                go.Scatter(
                    x=df_working["Qt1"],
                    y=df_working["depth"],
                    mode="lines",
                    name="Qt1",
                    line=dict(color="brown")
                ),
                row=1, col=4
            )
            
            # Columna 5: Ic SBTn
            fig.add_trace(
                go.Scatter(
                    x=df_working["Ic SBTn"],
                    y=df_working["depth"],
                    mode="lines",
                    name="Ic SBTn",
                    line=dict(color="purple")
                ),
                row=1, col=5
            )
            
            # Ejes y layout
            fig.update_yaxes(
                autorange="reversed",
                title="Profundidad (m)",
                row=1, col=1
            )
            fig.update_layout(
                height=800,
                width=1600,
                title="Perfiles de procesamiento inicial",
                showlegend=True
            )
            
            st.plotly_chart(fig)

else:
    st.info("Por favor, sube un archivo para comenzar.")
