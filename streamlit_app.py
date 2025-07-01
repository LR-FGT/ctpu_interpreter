import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

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

         # cross-correlation
        lags = signal.correlation_lags(len(df_checked["fs"]), len(df_checked["qc"]))
        ccf = ccf_values(df_checked["fs"], df_checked["qc"])
    
        # recortamos a lags -20 a 20
        valid_idx = np.where((lags >= -20) & (lags <= 20))[0]
        valid_lags = lags[valid_idx]
        valid_ccf = ccf[valid_idx]
    
        max_corr_idx = np.argmax(valid_ccf)
        max_lag = valid_lags[max_corr_idx]
    
        st.info(f"Lag con mayor correlación: {max_lag}")
    
        # plot
        fig = plot_ccf(valid_lags, valid_ccf, max_lag)
        st.plotly_chart(plot_ccf(valid_lags, valid_ccf, max_lag))
    
        # aplicar shift si el usuario quiere
        shift_ok = st.checkbox(
            f"¿Quieres aplicar shift con lag {max_lag}?", 
            value=True, 
            key="shift_confirm"
        )
        
    shift_applied = st.button("Aplicar shift")
    
    if shift_applied:
        if shift_ok and max_lag != 0:
            if max_lag < 0:
                df_checked = df_checked.iloc[abs(max_lag):].reset_index(drop=True)
            else:
                df_checked["fs"] = df_checked["fs"].shift(-max_lag)
                df_checked = df_checked.dropna().reset_index(drop=True)
            st.success("Shift aplicado.")
        else:
            st.info("No se aplicó shift.")

    # plot qc, fs, u2 con plotly
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=["qc (MPa)", "fs (MPa aprox)", "u2 (MPa aprox)"]
    )
    
    # qc
    fig.add_trace(
        go.Scatter(
            x=df_checked["qc"],
            y=df_checked["depth"],
            mode="lines",
            line=dict(color="blue"),
            name="qc"
        ),
        row=1, col=1
    )
    
    # fs
    fig.add_trace(
        go.Scatter(
            x=df_checked["fs"],
            y=df_checked["depth"],
            mode="lines",
            line=dict(color="red"),
            name="fs"
        ),
        row=1, col=2
    )
    
    # u2
    fig.add_trace(
        go.Scatter(
            x=df_checked["u2"],
            y=df_checked["depth"],
            mode="lines",
            line=dict(color="green"),
            name="u2"
        ),
        row=1, col=3
    )
    
    # Ajustes de ejes
    fig.update_yaxes(autorange="reversed", title="Profundidad (m)", row=1, col=1)
    fig.update_xaxes(title="qc (MPa)", row=1, col=1)
    fig.update_xaxes(title="fs (kPa)", row=1, col=2)
    fig.update_xaxes(title="u2 (kPa)", row=1, col=3)
    
    fig.update_layout(
        height=700,
        width=1200,
        title="Perfiles qc, fs y u2",
        showlegend=False
    )
    
    st.plotly_chart(fig)

    # En el siguiente paso conectaremos con gráficos
else:
    st.info("Por favor, sube un archivo para comenzar.")
