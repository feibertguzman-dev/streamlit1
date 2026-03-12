import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Académico y Financiero - Unilasallista",
    layout="wide"
)

# -----------------------------------------------------------------------------
# ENCABEZADO INSTITUCIONAL
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        # Intenta cargar el logo principal
        st.image("logoUnilasalle.png", width=200)
    except:
        st.write("*(Logo principal no encontrado)*")
        
with col_title:
    st.title("Dashboard Académico y Proyección Financiera")
    st.markdown("#### Corporación Universitaria Lasallista - Vicerrectoría Financiera")
st.markdown("---")

# -----------------------------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Leer el CSV asegurando el separador correcto
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    return df

try:
    df = load_data()
    
    # Asegurar que el Nivel sea numérico para los gráficos
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    
    # -----------------------------------------------------------------------------
    # BARRA LATERAL (SIDEBAR) - FILTROS
    # -----------------------------------------------------------------------------
    try:
        st.sidebar.image("est.png", use_column_width=True)
        st.sidebar.markdown("---")
    except:
        pass
        
    st.sidebar.header("Filtros Generales")
    
    programas = sorted(df['PROGRAMA'].dropna().unique())
    programa_seleccionado = st.sidebar.selectbox("Seleccionar Programa", ["Todos"] + list(programas))
    
    años = sorted(df['AÑO'].dropna().unique())
    año_seleccionado = st.sidebar.selectbox("Seleccionar Año", ["Todos"] + list(años))
    
    periodos = sorted(df['PeriodoAcadémico'].dropna().unique())
    periodo_seleccionado = st.sidebar.selectbox("Seleccionar Periodo Académico", ["Todos"] + list(periodos))
    
    estados = sorted(df['ESTADO'].dropna().unique())
    estado_seleccionado = st.sidebar.selectbox("Seleccionar Estado", ["Todos"] + list(estados))
    
    # Aplicar los filtros a una copia del dataframe
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
    tab1, tab2 = st.tabs(["Dashboard General", "Simulador de Pronósticos"])
    
    with tab1:
        st.subheader("Información General de la Población")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros (Filtrados)", f"{len(df_filtrado):,}")
        with col2:
            estudiantes_unicos = df_filtrado['DOCUMENTOIDENTIDAD'].nunique()
            st.metric("Estudiantes Únicos", f"{estudiantes_unicos:,}")
        with col3:
            st.metric("Programas Activos", df_filtrado['PROGRAMA'].nunique())
            
        with st.expander("Ver Tabla de Datos Filtrada"):
            st.dataframe(df_filtrado, use_container_width=True, height=300)
            
        st.markdown("---")
        
        # --- ANÁLISIS VISUAL PROFESIONAL (PLOTLY) ---
        st.markdown("### Análisis de Población Estudiantil")
        
        # Fila 1 de gráficos: Evolución y Composición
        colA, colB = st.columns(2)
        with colA:
            if not df_filtrado.empty:
                df_trend = df_filtrado.groupby(['PeriodoAcadémico', 'ESTADO']).size().reset_index(name='Cantidad')
                fig1 = px.area(
                    df_trend, x='PeriodoAcadémico', y='Cantidad', color='ESTADO',
                    title='Evolución de Estados por Periodo Académico',
                    template='plotly_white', markers=True
                )
                fig1.update_layout(legend_title="Estado", xaxis_title="Periodo", yaxis_title="N° Estudiantes")
                st.plotly_chart(fig1, use_container_width=True)

        with colB:
            if not df_filtrado.empty:
                # Filtrar valores nulos para el Sunburst
                df_sun = df_filtrado.dropna(subset=['ESTRATO', 'GENERO'])
                fig2 = px.sunburst(
                    df_sun, path=['ESTRATO', 'GENERO'], 
                    title='Composición Demográfica (Estrato y Género)',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        
        # Fila 2 de gráficos: Dispersión y Programas Top
        colC, colD = st.columns(2)
        with colC:
            if not df_filtrado.empty:
                fig3 = px.box(
                    df_filtrado, x='NIVEL', y='FACULTAD', color='GENERO',
                    title='Dispersión de Niveles por Facultad y Género',
                    template='plotly_white', points="all"
                )
                fig3.update_layout(xaxis_title="Nivel Cursado", yaxis_title="")
                st.plotly_chart(fig3, use_container_width=True)
                
        with colD:
            if not df_filtrado.empty:
                top_programas = df_filtrado['PROGRAMA'].value_counts().nlargest(10).index
                df_top_prog = df_filtrado[df_filtrado['PROGRAMA'].isin(top_programas)]
                df_estado_prog = df_top_prog.groupby(['PROGRAMA', 'ESTADO']).size().reset_index(name='Cantidad')
                fig4 = px.bar(
                    df_estado_prog, x='Cantidad', y='PROGRAMA', color='ESTADO', 
                    orientation='h', title='Top 10 Programas y su Estado Actual', 
                    barmode='stack', template='plotly_white'
                )
                fig4.update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title="", xaxis_title="N° Estudiantes")
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        
        # Fila 3: Análisis de Permanencia y Deserción
        st.markdown("### Análisis de Permanencia y Riesgo")
        colE, colF = st.columns(2)
        
        with colE:
            st.markdown("**Concentración de Estados por Nivel (Mapa de Calor)**")
            if not df_filtrado.empty:
                heat_data = df_filtrado.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Total')
                fig_heat = px.density_heatmap(
                    heat_data, x='NIVEL', y='ESTADO', z='Total',
                    color_continuous_scale='Blues', text_auto=True,
                    template='plotly_white'
                )
                fig_heat.update_layout(xaxis_title="Nivel Cursado", yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig_heat, use_container_width=True)

        with colF:
            st.markdown("**Distribución Porcentual de Estados por Facultad**")
            if not df_filtrado.empty:
                fig_prop = px.histogram(
                    df_filtrado, y="FACULTAD", color="ESTADO", 
                    barnorm="percent", orientation='h',
                    template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_prop.update_layout(xaxis_title="Proporción (%)", yaxis_title="", barmode='stack')
                st.plotly_chart(fig_prop, use_container_width=True)


    # -----------------------------------------------------------------------------
    # SIMULADOR DE PRONÓSTICOS
    # -----------------------------------------------------------------------------
    with tab2:
        st.header("Simulador de Pronósticos de Reingreso")
        st.markdown("""
        Este simulador evalúa el histórico de estudiantes (2021-1 a 2026-1) para proyectar posibles reingresos.
        Se enfoca **exclusivamente en estudiantes de Nivel 5 en adelante** que en el pasado han estado inactivos (Retirados/Cancelados).
        """)
        
        # Procesamiento de reglas
        df_pronostico = df.copy()
        df_sorted = df_pronostico.sort_values(by=['AÑO', 'PERIODO'])
        
        # Función para determinar si el estudiante reingresó históricamente
        def determinar_estado_final(estados):
            estados_list = estados.tolist()
            if 'Estudiante de Reingreso' in estados_list:
                return 'Reingresó'
            elif any(x in estados_list for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']):
                return 'Retirado/Cancelado'
            return 'Otro'
            
        estado_final = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(determinar_estado_final).reset_index(name='Estado_Histórico_Calculado')
        df_unique = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_final, on='DOCUMENTOIDENTIDAD')
        
        # Filtrar universo (Nivel >= 5)
        candidatos_potenciales = df_unique[(df_unique['NIVEL'] >= 5) & (df_unique['Estado_Histórico_Calculado'] == 'Retirado/Cancelado')]
        
        col_res1, col_res2 = st.columns(2)
        col_res1.info(f"Universo de estudiantes 'Retirados/Cancelados' a partir de Nivel 5: **{len(candidatos_potenciales)}**")
        
        # Simulación de Tasa
        tasa_recuperacion = col_res2.slider("Tasa de recuperación esperada (% por periodo)", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
        
        futuros_periodos = [f"{year}-{sem}" for year in range(2026, 2033) for sem in [1, 2]][1:13] 
        base_disponible = len(candidatos_potenciales)
        proyeccion = []
        
        for periodo in futuros_periodos:
            reingresos_estimados = int(base_disponible * tasa_recuperacion)
            proyeccion.append({
                'Periodo Académico': periodo, 
                'Reingresos Proyectados': reingresos_estimados, 
                'Población Restante por Contactar': base_disponible - reingresos_estimados
            })
            base_disponible -= reingresos_estimados
            
        df_proyeccion = pd.DataFrame(proyeccion)
        
        # Gráfica de proyección
        st.markdown("### Proyección Semestral de Reingresos")
        fig_proy = px.bar(
            df_proyeccion, x='Periodo Académico', y='Reingresos Proyectados',
            text_auto=True, template='plotly_white', color_discrete_sequence=['#1f77b4']
        )
        fig_proy.update_layout(yaxis_title="Cantidad de Estudiantes", xaxis_title="")
        st.plotly_chart(fig_proy, use_container_width=True)
        
        with st.expander("Ver Tabla Detallada de la Proyección"):
            st.dataframe(df_proyeccion, use_container_width=True)
        
        # Exportación a Power BI
        st.markdown("### Exportación de Datos para Power BI")
        st.markdown("Descarga la base de datos limpia y procesada para integrarla directamente a tu tablero en Power BI.")
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            candidatos_potenciales.to_excel(writer, sheet_name='Base_Target_Nivel5+', index=False)
            df_proyeccion.to_excel(writer, sheet_name='Pronostico_2026_2032', index=False)
        output.seek(0)
        
        st.download_button(
            label="Descargar Archivo para Power BI (.xlsx)",
            data=output,
            file_name="Datos_Pronostico_Reingresos_Unilasallista.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
except Exception as e:
    st.error("Error al procesar los datos.")
    st.exception(e)
    st.info("Asegúrate de tener instaladas las librerías: streamlit, pandas, numpy, plotly, openpyxl, y xlsxwriter.")
