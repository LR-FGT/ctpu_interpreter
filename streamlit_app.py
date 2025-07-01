import streamlit as st
import pandas as pd

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
    
    # conversión de unidades
    st.subheader("Conversión de unidades")

    qc_in_mpa = st.radio("¿La resistencia de punta qc está en MPa?", options=["Sí", "No"])
    fs_in_kpa = st.radio("¿La fricción fs está en kPa?", options=["Sí", "No"])
    u2_in_kpa = st.radio("¿La presión u2 está en kPa?", options=["Sí", "No"])

    if st.button("Procesar archivo"):
        st.info("En el próximo paso conectaremos este botón con el procesamiento completo.")

else:
    st.info("Por favor, sube un archivo para comenzar.")


