import streamlit as st
import pandas as pd

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
       st.subheader("Procesando archivo...")

        # aplicar conversión de unidades
        if qc_in_mpa == "No":
            factor_qc = st.number_input("Factor de conversión para qc:", value=1.0)
            df["qc"] = df["qc"] * factor_qc
        if fs_in_kpa == "No":
            factor_fs = st.number_input("Factor de conversión para fs:", value=1.0)
            df["fs"] = df["fs"] * factor_fs
        if u2_in_kpa == "No":
            factor_u2 = st.number_input("Factor de conversión para u2:", value=1.0)
            df["u2"] = df["u2"] * factor_u2
        
        # ejecutar el chequeo de consistencia
        df_checked = sanity_check(df)
        
        st.success("Sanity check aplicado correctamente. Muestra del resultado:")
        st.dataframe(df_checked.head())
    
        # En el siguiente paso conectaremos con gráficos
else:
    st.info("Por favor, sube un archivo para comenzar.")


