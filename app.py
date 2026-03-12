import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analítico y Predictivo - Unilasallista",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ENCABEZADO INSTITUCIONAL
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        st.image("logoUnilasalle.png", width=180)
    except:
        st.write("*(Logo principal no encontrado)*")
        
with col_title:
    st.title("Sistema Integrado de Análisis y Proyección de Reingresos")
    st.markdown("#### Corporación Universitaria Lasallista - Vicerrectoría Financiera")
st.markdown("---")

# -----------------------------------------------------------------------------
# CARGA DE DATOS Y CACHÉ
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    return df

try:
    df = load_data()
    
    # -----------------------------------------------------------------------------
    # BARRA LATERAL (SIDEBAR) - FILTROS GLOBALES
    # -----------------------------------------------------------------------------
    try:
        st.sidebar.image("est.png", use_column_width=True)
        st.sidebar.markdown("---")
    except:
        pass
        
    st.sidebar.header("Filtros Globales (Dashboard)")
    st.sidebar.markdown("*(Aplican a la pestaña de Dashboard General)*")
    
    programas = sorted(df['PROGRAMA'].dropna().unique())
    programa_seleccionado = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(programas))
    
    años = sorted(df['AÑO'].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("Año", ["Todos"] + list(años))
    
    periodos = sorted(df['PeriodoAcadémico'].dropna().unique())
    periodo_seleccionado = st.sidebar.selectbox("Periodo Académico", ["Todos"] + list(periodos))
    
    estados = sorted(df['ESTADO'].dropna().unique())
    estado_seleccionado = st.sidebar.selectbox("Estado Actual", ["Todos"] + list(estados))
    
    # Aplicar filtros globales
    df_filtrado = df.copy()
    if programa_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['PROGRAMA'] == programa_seleccionado]
    if año_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['AÑO'] == año_seleccionado]
    if periodo_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['PeriodoAcadémico'] == periodo_seleccionado]
    if estado_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['ESTADO'] == estado_seleccionado]

    # -----------------------------------------------------------------------------
    # PESTAÑAS PRINCIPALES
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard General", 
        "⚙️ Simulador Interactivo (Escenarios)", 
        "🤖 Pronóstico Automático (Machine Learning)",
        "📖 Documentación y Guía de Uso"
    ])
    
    # =============================================================================
    # PESTAÑA 1: DASHBOARD GENERAL
    # =============================================================================
    with tab1:
        st.subheader("Análisis Demográfico y Académico Histórico")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Registros (Filtrados)", f"{len(df_filtrado):,}")
        col2.metric("Estudiantes Únicos", f"{df_filtrado['DOCUMENTOIDENTIDAD'].nunique():,}")
        col3.metric("Programas Activos", df_filtrado['PROGRAMA'].nunique())
            
        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            if not df_filtrado.empty:
                df_trend = df_filtrado.groupby(['PeriodoAcadémico', 'ESTADO']).size().reset_index(name='Cantidad')
                fig1 = px.area(df_trend, x='PeriodoAcadémico', y='Cantidad', color='ESTADO',
                               title='Evolución Histórica de Estados por Periodo', template='plotly_white')
                st.plotly_chart(fig1, use_container_width=True)

        with colB:
            if not df_filtrado.empty:
                heat_data = df_filtrado.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Total')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Total',
                                              color_continuous_scale='Blues', text_auto=True,
                                              title="Mapa de Calor: Riesgo de Deserción por Nivel",
                                              template='plotly_white')
                st.plotly_chart(fig_heat, use_container_width=True)

    # =============================================================================
    # PREPARACIÓN DE DATOS PARA SIMULADOR Y ML (Target: Nivel >= 5, Retirados)
    # =============================================================================
    df_modelado = df.copy()
    df_sorted_mod = df_modelado.sort_values(by=['AÑO', 'PERIODO'])
    
    def determinar_estado_real(estados):
        lista = estados.tolist()
        if 'Estudiante de Reingreso' in lista:
            return 'Reingresó Históricamente'
        elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']):
            return 'Candidato a Reingreso'
        return 'No Aplica'
        
    estado_final_mod = df_sorted_mod.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(determinar_estado_real).reset_index(name='Estado_Campaña')
    df_unique_mod = df_sorted_mod.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_final_mod, on='DOCUMENTOIDENTIDAD')
    
    # Universo total a contactar
    candidatos_base = df_unique_mod[(df_unique_mod['NIVEL'] >= 5) & (df_unique_mod['Estado_Campaña'] == 'Candidato a Reingreso')]
    
    # =============================================================================
    # PESTAÑA 2: SIMULADOR INTERACTIVO
    # =============================================================================
    with tab2:
        st.header("Simulador de Escenarios (Modelo Determinístico)")
        st.info("Utiliza esta herramienta para simular estrategias financieras evaluando qué pasaría si se logra una tasa de recuperación específica mediante campañas manuales o incentivos.")
        
        col_fil1, col_fil2 = st.columns(2)
        facultad_sim = col_fil1.selectbox("Segmentar Target por Facultad", ["Todas"] + list(sorted(df['FACULTAD'].dropna().unique())))
        
        progs_disp = sorted(df[df['FACULTAD'] == facultad_sim]['PROGRAMA'].dropna().unique()) if facultad_sim != "Todas" else sorted(df['PROGRAMA'].dropna().unique())
        programa_sim = col_fil2.selectbox("Segmentar Target por Programa", ["Todos"] + list(progs_disp))
        
        # Filtro de candidatos
        candidatos_sim = candidatos_base.copy()
        if facultad_sim != "Todas": candidatos_sim = candidatos_sim[candidatos_sim['FACULTAD'] == facultad_sim]
        if programa_sim != "Todos": candidatos_sim = candidatos_sim[candidatos_sim['PROGRAMA'] == programa_sim]
            
        base_inicial_sim = len(candidatos_sim)
        st.markdown("---")
        
        tasa_recuperacion = st.slider("🎯 Tasa de éxito esperada por campaña (% por periodo)", min_value=1.0, max_value=40.0, value=15.0, step=1.0) / 100.0
        
        futuros_periodos = [f"{year}-{sem}" for year in range(2026, 2033) for sem in [1, 2]][1:13] 
        base_disp = base_inicial_sim
        proyeccion_sim = []
        
        for per in futuros_periodos:
            reing = int(round(base_disp * tasa_recuperacion)) if base_disp > 0 else 0
            if reing > base_disp: reing = base_disp
            proyeccion_sim.append({'Periodo': per, 'Reingresos Estimados': reing, 'Base Restante': base_disp - reing})
            base_disp -= reing
            
        df_proy_sim = pd.DataFrame(proyeccion_sim)
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("👥 Población Target", f"{base_inicial_sim:,}")
        col_kpi2.metric("📈 Total Reingresos Simulado", f"{df_proy_sim['Reingresos Estimados'].sum():,}")
        col_kpi3.metric("💰 Retorno de Campaña", f"{(df_proy_sim['Reingresos Estimados'].sum() / base_inicial_sim * 100) if base_inicial_sim > 0 else 0:.1f}%")
        
        fig_sim = px.bar(df_proy_sim, x='Periodo', y='Reingresos Estimados', text_auto=True, template='plotly_white', title="Curva de Recuperación Simulada", color_discrete_sequence=['#17becf'])
        st.plotly_chart(fig_sim, use_container_width=True)

    # =============================================================================
    # PESTAÑA 3: PRONÓSTICO CON MACHINE LEARNING
    # =============================================================================
    with tab3:
        st.header("Pronóstico Estadístico con Machine Learning")
        st.info("Este módulo utiliza un algoritmo de **Regresión Lineal (Scikit-Learn)** entrenado con el comportamiento histórico (2021-2026) para predecir de forma automática la tendencia de reingresos orgánicos futuros, sin intervención manual de tasas.")
        
        # Preparar datos de entrenamiento (Histórico de reingresos reales por periodo)
        reingresos_hist = df_unique_mod[df_unique_mod['Estado_Campaña'] == 'Reingresó Históricamente']
        tendencia_real = df_sorted_mod[df_sorted_mod['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos_Reales')
        
        if len(tendencia_real) > 2:
            # Crear índice temporal numérico para ML
            tendencia_real['TimeIndex'] = range(1, len(tendencia_real) + 1)
            X_train = tendencia_real[['TimeIndex']]
            y_train = tendencia_real['Reingresos_Reales']
            
            # Entrenar el modelo de ML
            ml_model = LinearRegression()
            ml_model.fit(X_train, y_train)
            
            # Generar predicciones futuras
            ultimo_indice = tendencia_real['TimeIndex'].max()
            X_future = pd.DataFrame({'TimeIndex': range(ultimo_indice + 1, ultimo_indice + 1 + len(futuros_periodos))})
            predicciones_ml = ml_model.predict(X_future)
            
            # Limpiar predicciones (no pueden ser negativas, redondear a enteros)
            predicciones_ml = [max(0, int(round(p))) for p in predicciones_ml]
            
            df_proy_ml = pd.DataFrame({
                'Periodo Académico': futuros_periodos,
                'Reingresos Proyectados (ML)': predicciones_ml
            })
            
            col_ml1, col_ml2 = st.columns(2)
            col_ml1.metric("🤖 Modelo Utilizado", "Linear Regression (OLS)")
            col_ml2.metric("📈 Total Reingresos Automáticos (2026-2032)", f"{sum(predicciones_ml)}")
            
            fig_ml = px.line(df_proy_ml, x='Periodo Académico', y='Reingresos Proyectados (ML)', markers=True, 
                             title="Pronóstico Predictivo de Reingresos Orgánicos (Tendencia AI)",
                             template="plotly_white", color_discrete_sequence=['#ff7f0e'])
            fig_ml.update_traces(line=dict(width=3))
            st.plotly_chart(fig_ml, use_container_width=True)
            
            with st.expander("Ver Datos de Entrenamiento vs Predicción"):
                st.write("Histórico de Entrenamiento:")
                st.dataframe(tendencia_real[['PeriodoAcadémico', 'Reingresos_Reales']].T)
                st.write("Predicción del Modelo:")
                st.dataframe(df_proy_ml.T)
        else:
            st.warning("No hay suficientes datos históricos de reingreso en el CSV actual para entrenar el modelo de Machine Learning. El algoritmo necesita al menos 3 periodos con reingresos.")

    # =============================================================================
    # PESTAÑA 4: DOCUMENTACIÓN Y GUÍA DE USO
    # =============================================================================
    with tab4:
        st.header("📖 Documentación Oficial del Sistema")
        
        st.markdown("""
        ### Arquitectura del Dashboard
        Este sistema fue diseñado para la Vicerrectoría Financiera de la **Corporación Universitaria Lasallista**. Integra técnicas de inteligencia de negocios (BI), simulaciones financieras y Machine Learning para maximizar la retención y recuperación de estudiantes.
        
        ---
        
        #### 1. ¿Cómo se procesan los datos? (Reglas de Negocio)
        El sistema depura automáticamente la base de datos `DataSPSSReingreso.csv` para encontrar oportunidades financieras viables:
        * **Filtro de Nivel Académico:** Excluye a estudiantes de los niveles 1 al 4. Se concentra exclusivamente en estudiantes del **Nivel 5 en adelante**, ya que estadísticamente representan un menor riesgo de deserción secundaria y tienen mayor motivación para terminar su carrera.
        * **Detección de Estado Real:** El código agrupa los registros históricos de cada documento de identidad. Si un estudiante fue clasificado como "Retirado" en 2022, pero aparece como "Estudiante de Reingreso" en 2024, el algoritmo lo clasifica automáticamente como un caso de éxito histórico y lo saca de la base de target actual.
        
        #### 2. Diferencia entre el Simulador y el Pronóstico ML
        * **Simulador Interactivo (Pestaña 2):** Utiliza un enfoque *Gamificado / Determinístico*. Permite a los directivos jugar con distintos escenarios (Ej: *"¿Qué pasa si hacemos una campaña de llamadas y logramos un 20% de conversión en la Facultad de Ingeniería?"*). Es ideal para establecer metas y presupuestos para el equipo de admisiones.
        * **Pronóstico con ML (Pestaña 3):** Utiliza Inteligencia Artificial (específicamente un algoritmo de *Scikit-Learn: Linear Regression*). En lugar de preguntar "¿Qué tasa queremos?", el algoritmo analiza matemáticamente la línea de tendencia histórica (2021-2026) y predice el comportamiento futuro orgánico de la población estudiantil asumiendo que las condiciones se mantienen.
        
        #### 3. Mantenimiento y Actualización de Datos
        Actualizar este tablero es un proceso continuo que **no requiere conocimientos de programación**.
        1. Al final del periodo académico, extrae tu nueva sábana de datos del sistema de la universidad.
        2. Asegúrate de que las columnas tengan los mismos nombres que la estructura original.
        3. Guarda el archivo como **`DataSPSSReingreso.csv`**.
        4. Reemplaza este archivo en la carpeta principal de tu proyecto (o súbelo a tu repositorio de GitHub).
        5. El sistema leerá los nuevos datos, actualizará todos los gráficos y reentrenará automáticamente el modelo de Machine Learning en milisegundos.
        """)
        
        st.markdown("---")
        st.markdown("### 📥 Descarga de Estructuras para Power BI")
        st.info("Genera el set de datos unificado (Histórico + Proyecciones) para enlazarlo como origen en la suite de Power BI corporativa.")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            candidatos_base.to_excel(writer, sheet_name='Base_Target_Nivel5+', index=False)
            df_proy_sim.to_excel(writer, sheet_name='Proyeccion_Escenarios', index=False)
        output.seek(0)
        st.download_button(
            label="📊 Exportar Base de Datos a Excel (.xlsx)",
            data=output,
            file_name="Data_Unificada_Unilasallista.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
except Exception as e:
    st.error("Error crítico al inicializar el sistema.")
    st.exception(e)
