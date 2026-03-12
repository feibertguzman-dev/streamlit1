import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Estudiantes - Unilasallista",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Dashboard de Estudiantes - Universidad La Salle")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    return df

# Cargar datos
try:
    df = load_data()
    
    # Crear Pestañas
    tab1, tab2 = st.tabs(["📋 Dashboard General", "📈 Simulador de Pronósticos (Vicerrectoría Financiera)"])
    
    with tab1:
        # Información básica del dataset
        st.header("📋 Información General")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", len(df))
        with col2:
            st.metric("Número de Columnas", len(df.columns))
        with col3:
            st.metric("Programas Únicos", df['PROGRAMA'].nunique())
        
        st.markdown("---")
        
       # Sidebar con filtros
        st.sidebar.header("🔍 Filtros Generales")
        
        programas = df['PROGRAMA'].unique()
        programa_seleccionado = st.sidebar.selectbox("Seleccionar Programa", ["Todos"] + list(programas))
        
        años = sorted(df['AÑO'].unique())
        año_seleccionado = st.sidebar.selectbox("Seleccionar Año", ["Todos"] + list(años))
        
        # --- NUEVO FILTRO DE PERIODO ACADÉMICO ---
        # Extraemos los periodos únicos (ej. 2021-1, 2021-2) y los ordenamos
        periodos = sorted(df['PeriodoAcadémico'].dropna().unique())
        periodo_seleccionado = st.sidebar.selectbox("Seleccionar Periodo Académico", ["Todos"] + list(periodos))
        
        estados = df['ESTADO'].unique()
        estado_seleccionado = st.sidebar.selectbox("Seleccionar Estado", ["Todos"] + list(estados))
        
        # Aplicar filtros
        df_filtrado = df.copy()
        if programa_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['PROGRAMA'] == programa_seleccionado]
            
        if año_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['AÑO'] == año_seleccionado]
            
        # Aplicar la lógica del nuevo filtro
        if periodo_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['PeriodoAcadémico'] == periodo_seleccionado]
            
        if estado_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['ESTADO'] == estado_seleccionado]
        
        st.header(f"📊 Datos Filtrados ({len(df_filtrado)} registros)")
        st.dataframe(df_filtrado, use_container_width=True, height=400)
        st.markdown("---")
        
        # Estadísticas y gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👥 Distribución por Género")
            genero_counts = df_filtrado['GENERO'].value_counts()
            fig1, ax1 = plt.subplots()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax1.pie(genero_counts.values, labels=genero_counts.index, autopct='%1.1f%%', colors=colors[:len(genero_counts)])
            ax1.axis('equal')
            st.pyplot(fig1)
        
        with col2:
            st.subheader("🏢 Distribución por Estado")
            estado_counts = df_filtrado['ESTADO'].value_counts()
            fig2, ax2 = plt.subplots()
            sns.barplot(x=estado_counts.values, y=estado_counts.index, ax=ax2, palette="viridis")
            ax2.set_xlabel("Cantidad")
            ax2.set_ylabel("Estado")
            st.pyplot(fig2)
        
        st.markdown("---")
        
        # Distribución por estrato y año
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("🏠 Distribución por Estrato")
            estrato_counts = df_filtrado['ESTRATO'].value_counts()
            fig5, ax5 = plt.subplots()
            sns.barplot(x=estrato_counts.values, y=estrato_counts.index, ax=ax5, palette="Blues_d")
            ax5.set_xlabel("Cantidad de Estudiantes")
            ax5.set_ylabel("Estrato")
            st.pyplot(fig5)
        
        with col4:
            st.subheader("📅 Estudiantes por Año")
            año_counts = df_filtrado['AÑO'].value_counts().sort_index()
            fig6, ax6 = plt.subplots()
            sns.lineplot(x=año_counts.index, y=año_counts.values, marker='o', ax=ax6, linewidth=2)
            ax6.set_xlabel("Año")
            ax6.set_ylabel("Cantidad de Estudiantes")
            ax6.set_xticks(año_counts.index)
            st.pyplot(fig6)

    with tab2:
        st.header("📈 Simulador de Pronósticos de Reingreso")
        st.markdown("""
        Este simulador evalúa los históricos (2021-1 a 2026-1) para proyectar cuántos estudiantes inactivos o retirados del **Nivel 5 en adelante** tienen probabilidad de reingreso desde el periodo **2026-2 hasta 2032-1**.
        """)
        
        # Procesamiento de reglas de negocio para pronóstico
        df_pronostico = df.copy()
        df_pronostico['NIVEL'] = pd.to_numeric(df_pronostico['NIVEL'], errors='coerce').fillna(0)
        
        # 1. Agrupar por estudiante para conocer si alguna vez se retiró y luego reingresó
        # Ordenamos temporalmente
        df_sorted = df_pronostico.sort_values(by=['AÑO', 'PERIODO'])
        
        def determinar_estado_final(estados):
            estados_list = estados.tolist()
            if 'Estudiante de Reingreso' in estados_list:
                return 'Reingresó'
            elif any(x in estados_list for x in ['Estudiante Retirado', 'Canceló Periodo']):
                return 'Retirado/Cancelado'
            return 'Otro'
            
        estado_final = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(determinar_estado_final).reset_index(name='Estado_Histórico_Calculado')
        df_unique = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_final, on='DOCUMENTOIDENTIDAD')
        
        # 2. Filtrar ventana (Nivel >= 5) y candidatos a reingresar (Retirados/Cancelados en el pasado)
        candidatos_potenciales = df_unique[(df_unique['NIVEL'] >= 5) & (df_unique['Estado_Histórico_Calculado'] == 'Retirado/Cancelado')]
        reingresos_historicos = df_unique[(df_unique['NIVEL'] >= 5) & (df_unique['Estado_Histórico_Calculado'] == 'Reingresó')]
        
        st.info(f"Según los datos históricos filtrados (Nivel >= 5), se encontraron **{len(candidatos_potenciales)}** estudiantes en estado 'Retirado/Cancelado' que son target de contacto para reingreso.")
        
        # 3. Modelo simple de proyección de reingresos
        # Generar periodos de 2026-2 a 2032-1
        futuros_periodos = [f"{year}-{sem}" for year in range(2026, 2033) for sem in [1, 2]][1:13] 
        
        # Simulamos una tendencia de reingresos basada en una tasa de recuperación objetivo (por ejemplo, 15% por periodo)
        # Puedes ajustar este valor desde el sidebar de la pestaña
        tasa_recuperacion = st.slider("Tasa de recuperación estimada (% por periodo)", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
        
        base_disponible = len(candidatos_potenciales)
        proyeccion = []
        for periodo in futuros_periodos:
            reingresos_estimados = int(base_disponible * tasa_recuperacion)
            proyeccion.append({'Periodo': periodo, 'Reingresos Proyectados': reingresos_estimados, 'Base Restante por Contactar': base_disponible})
            base_disponible -= reingresos_estimados  # Se restan los que reingresan de la base
            
        df_proyeccion = pd.DataFrame(proyeccion)
        
        # Mostrar gráfica de proyección
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_proyeccion, x='Periodo', y='Reingresos Proyectados', palette="crest", ax=ax7)
        ax7.set_title("Proyección de Reingresos Esperados (Nivel 5+)")
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45)
        st.pyplot(fig7)
        
        st.dataframe(df_proyeccion, use_container_width=True)
        
        # 4. Exportar a Power BI
        st.markdown("### 📥 Exportación para Power BI")
        st.markdown("Descarga la tabla procesada con los campos calculados de los estudiantes target (Nivel >= 5) para alimentar el dashboard en Power BI.")
        
        # Preparar Excel en memoria
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            candidatos_potenciales.to_excel(writer, sheet_name='Base_Estudiantes_Target', index=False)
            df_proyeccion.to_excel(writer, sheet_name='Pronostico_2026_2032', index=False)
        output.seek(0)
        
        st.download_button(
            label="Descargar Tabla para Power BI (.xlsx)",
            data=output,
            file_name="Datos_Pronostico_Reingresos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.info("Asegúrate de que el archivo DataSPSSReingreso.csv esté en el mismo directorio que app.py")

