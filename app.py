import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# CONFIGURACIÓN CORPORATIVA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inteligencia de Negocios - Unilasallista", layout="wide", initial_sidebar_state="expanded")

# Configuración de Seaborn para un look profesional
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

col_logo, col_title = st.columns([1, 4])
with col_logo:
    try: st.image("logoUnilasalle.png", width=180)
    except: st.write("*(Logo no encontrado)*")
with col_title:
    st.title("Inteligencia Analítica y Proyección de Reingresos")
    st.markdown("#### Vicerrectoría Financiera | Corporación Universitaria Lasallista")
st.markdown("---")

# -----------------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    df['ESTRATO_NUM'] = df['ESTRATO'].str.extract('(\d+)').astype(float).fillna(0)
    df['CIUDADRESIDENCIA'] = df['CIUDADRESIDENCIA'].astype(str).str.upper().str.strip()
    return df

try:
    df = load_data()
    
    # -----------------------------------------------------------------------------
    # PANEL IZQUIERDO (SIDEBAR) - FILTROS INTELIGENTES
    # -----------------------------------------------------------------------------
    try:
        st.sidebar.image("est.png", use_column_width=True)
    except: pass
    
    st.sidebar.markdown("### ⚙️ Parámetros de Segmentación")
    st.sidebar.info("Estos filtros aplican al análisis histórico, al simulador y a los modelos predictivos simultáneamente.")
    
    # Filtros Demográficos (Aplican a TODO)
    facultades = sorted(df['FACULTAD'].dropna().unique())
    fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(facultades))
    
    progs_disp = sorted(df[df['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df['PROGRAMA'].dropna().unique())
    prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
    
    estratos = sorted(df['ESTRATO'].dropna().unique())
    est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(estratos))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Filtro Temporal (Solo Vista General)")
    periodos = sorted(df['PeriodoAcadémico'].dropna().unique())
    per_sel = st.sidebar.selectbox("Periodo Específico", ["Histórico Completo"] + list(periodos))

    # APLICACIÓN DE FILTROS AL DATAFRAME PRINCIPAL
    df_base = df.copy()
    if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
    if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
    if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]

    # Mensaje Dinámico de Contexto
    st.success(f"📌 **Contexto Actual:** Analizando base de datos para **{'Todas las Facultades' if fac_sel == 'Todas' else fac_sel}** "
               f"y **{'Todos los Programas' if prog_sel == 'Todos' else prog_sel}**. "
               f"Población total filtrada: **{df_base['DOCUMENTOIDENTIDAD'].nunique():,} estudiantes únicos**.")

    # -----------------------------------------------------------------------------
    # PESTAÑAS DEL SISTEMA
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis Descriptivo & Geoespacial", "⚙️ Simulador de Impacto", "🧠 Modelos Machine Learning", "📖 Documentación"])
    
    # =============================================================================
    # PESTAÑA 1: DASHBOARD Y MAPAS (SEABORN + PLOTLY MAP)
    # =============================================================================
    with tab1:
        # Aplicamos el filtro temporal solo para esta vista
        df_vista = df_base.copy()
        if per_sel != "Histórico Completo":
            df_vista = df_vista[df_vista['PeriodoAcadémico'] == per_sel]
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros Totales", f"{len(df_vista):,}")
        col2.metric("Matriculados Activos", f"{len(df_vista[df_vista['ESTADO'] == 'Estudiante Matriculado']):,}")
        col3.metric("Retirados / Aplazados", f"{len(df_vista[df_vista['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado'])]):,}")
        col4.metric("Promedio Nivel", f"{df_vista['NIVEL'].mean():.1f}")
        
        st.markdown("---")
        
        # Fila 1 de Gráficos Seaborn
        st.markdown("### 📉 Diagnóstico de Permanencia")
        figA, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafico 1: Estado por Nivel Académico
        sns.boxplot(data=df_vista, x='ESTADO', y='NIVEL', hue='GENERO', ax=ax1, palette="Set2")
        ax1.set_title("Distribución de Estados Académicos por Nivel", fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_xlabel("")
        ax1.set_ylabel("Nivel Cursado")
        
        # Grafico 2: Concentración de Estratos
        sns.countplot(data=df_vista, y='ESTRATO', hue='ESTADO', ax=ax2, palette="viridis", order=sorted(df_vista['ESTRATO'].dropna().unique()))
        ax2.set_title("Impacto Socioeconómico (Estrato vs Estado)", fontweight='bold')
        ax2.set_xlabel("Cantidad de Estudiantes")
        ax2.set_ylabel("")
        
        st.pyplot(figA)
        
        # MAPA GEOESPACIAL DE ESTUDIANTES
        st.markdown("---")
        st.markdown("### 📍 Distribución Geográfica del Mercado Estudiantil")
        
        # Diccionario de coordenadas para ciudades principales (Antioquia/Colombia)
        coords = {
            'MEDELLIN': (6.2442, -75.5812), 'CALDAS': (6.0911, -75.6383), 'ENVIGADO': (6.1759, -75.5917),
            'ITAGUI': (6.1718, -75.6095), 'SABANETA': (6.1515, -75.6166), 'LA ESTRELLA': (6.1576, -75.6443),
            'BELLO': (6.3373, -75.5579), 'COPACABANA': (6.3463, -75.5089), 'GIRARDOTA': (6.3768, -75.4457),
            'AMAGA': (6.0385, -75.7034), 'BOGOTA': (4.7110, -74.0721), 'RIONEGRO': (6.1551, -75.3737)
        }
        
        df_geo = df_vista.groupby('CIUDADRESIDENCIA').size().reset_index(name='Concentración')
        df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
        df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
        df_geo = df_geo.dropna(subset=['Lat']) # Solo mapeamos las que tienen coordenadas
        
        if not df_geo.empty:
            fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Concentración", color="Concentración",
                                        hover_name="CIUDADRESIDENCIA", color_continuous_scale="Viridis",
                                        size_max=50, zoom=9, mapbox_style="carto-positron",
                                        title="Zonas de mayor densidad estudiantil (Basado en Ciudad de Residencia)")
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No hay suficientes datos de ciudades reconocidas en este filtro para generar el mapa.")

    # =============================================================================
    # MOTOR DE DATOS PARA SIMULADOR Y ML (Aplica filtros globales)
    # =============================================================================
    df_sorted_mod = df_base.sort_values(by=['AÑO', 'PERIODO'])
    
    def clasificar_target(estados):
        lista = estados.tolist()
        if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
        elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingreso'
        return 'No Aplica'
        
    estado_calc = df_sorted_mod.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Estado_Campaña')
    df_univ = df_sorted_mod.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
    candidatos_base = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Estado_Campaña'] == 'Candidato a Reingreso')]

    # =============================================================================
    # PESTAÑA 2: SIMULADOR DE IMPACTO FINANCIERO
    # =============================================================================
    with tab2:
        st.header("Simulador de Retorno de Estrategias")
        st.info("La gráfica inferior ha sido rediseñada para ser formal: Las barras representan las nuevas captaciones (Reingresos) y la línea roja representa cómo se agota la base de datos de estudiantes disponibles.")
        
        base_inicial_sim = len(candidatos_base)
        tasa_recup = st.slider("🎯 Tasa de conversión de llamadas/campañas (%)", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
        
        per_futuros = [f"{year}-{sem}" for year in range(2026, 2033) for sem in [1, 2]][1:13] 
        base_disp = base_inicial_sim
        proy_sim = []
        
        for per in per_futuros:
            reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
            if reing > base_disp: reing = base_disp
            base_disp -= reing
            proy_sim.append({'Periodo': per, 'Reingresos Nuevos': reing, 'Inventario Pendiente': base_disp})
            
        df_proy_sim = pd.DataFrame(proy_sim)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Universo Filtrado (Nivel 5+)", f"{base_inicial_sim:,}")
        c2.metric("Reingresos Simulados Totales", f"{df_proy_sim['Reingresos Nuevos'].sum():,}")
        c3.metric("Inventario Agotado", f"{base_inicial_sim - df_proy_sim['Inventario Pendiente'].iloc[-1]:,}")
        
        # Grafico combinado Seaborn/Matplotlib Profesional
        figB, ax_bar = plt.subplots(figsize=(10, 4))
        ax_line = ax_bar.twinx()
        
        sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos Nuevos', ax=ax_bar, color='#4C72B0', alpha=0.8, label="Nuevos Reingresos")
        sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario Pendiente', ax=ax_line, color='#C44E52', marker='o', linewidth=2.5, label="Base Pendiente por Contactar")
        
        ax_bar.set_title("Proyección Financiera y Agotamiento de Base", fontweight='bold')
        ax_bar.set_ylabel("Cant. Estudiantes Recuperados")
        ax_line.set_ylabel("Inventario Restante", color='#C44E52')
        ax_bar.tick_params(axis='x', rotation=45)
        
        # Leyendas unificadas
        lines_1, labels_1 = ax_bar.get_legend_handles_labels()
        lines_2, labels_2 = ax_line.get_legend_handles_labels()
        ax_bar.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        ax_line.get_legend().remove()
        
        st.pyplot(figB)

    # =============================================================================
    # PESTAÑA 3: MACHINE LEARNING (REGRESIÓN Y ÁRBOL DE DECISIÓN)
    # =============================================================================
    with tab3:
        st.header("Inteligencia Artificial y Machine Learning")
        
        ml1, ml2 = st.tabs(["🌳 Perfilamiento (Árbol de Decisión)", "📈 Proyección (Regresión Lineal)"])
        
        with ml1:
            st.markdown("### Perfil de Probabilidad de Reingreso")
            st.write("Este modelo de Clasificación (Decision Tree) analiza qué variables pesan más para que un estudiante retirado decida volver.")
            
            # Preparar datos para el árbol
            df_tree = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Estado_Campaña'].isin(['Reingresó Históricamente', 'Candidato a Reingreso']))].copy()
            df_tree['Target'] = np.where(df_tree['Estado_Campaña'] == 'Reingresó Históricamente', 1, 0)
            
            if len(df_tree) > 10 and df_tree['Target'].sum() > 0:
                features = ['NIVEL', 'ESTRATO_NUM']
                X = df_tree[features].fillna(0)
                y = df_tree['Target']
                
                clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
                clf.fit(X, y)
                
                # Gráfico de Importancia de Variables
                importances = pd.DataFrame({'Variable': features, 'Importancia': clf.feature_importances_}).sort_values(by='Importancia', ascending=False)
                figC, ax_tree1 = plt.subplots(figsize=(8, 3))
                sns.barplot(data=importances, x='Importancia', y='Variable', ax=ax_tree1, palette="mako")
                ax_tree1.set_title("¿Qué variables influyen más en el reingreso?", fontweight='bold')
                st.pyplot(figC)
                
                # Visualización del Árbol
                st.markdown("**Reglas de Decisión del Algoritmo:**")
                figD, ax_tree2 = plt.subplots(figsize=(12, 6), dpi=300)
                plot_tree(clf, feature_names=features, class_names=['No Vuelve', 'Reingresa'], filled=True, rounded=True, ax=ax_tree2, proportion=True)
                st.pyplot(figD)
            else:
                st.warning("⚠️ No hay suficientes casos de reingreso histórico bajo los filtros actuales para entrenar el Árbol de Decisión. Prueba quitando el filtro de Facultad.")

        with ml2:
            st.markdown("### Tendencia Orgánica Futura")
            reingresos_hist = df_univ[df_univ['Estado_Campaña'] == 'Reingresó Históricamente']
            tendencia = df_sorted_mod[df_sorted_mod['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos')
            
            if len(tendencia) > 2:
                tendencia['Time'] = range(1, len(tendencia) + 1)
                modelo = LinearRegression()
                modelo.fit(tendencia[['Time']], tendencia['Reingresos'])
                
                T_future = pd.DataFrame({'Time': range(tendencia['Time'].max() + 1, tendencia['Time'].max() + 1 + len(per_futuros))})
                preds = [max(0, int(round(p))) for p in modelo.predict(T_future)]
                
                df_preds = pd.DataFrame({'Periodo': per_futuros, 'Reingresos Proyectados (ML)': preds})
                
                figE, ax_reg = plt.subplots(figsize=(10, 4))
                sns.regplot(data=tendencia, x='Time', y='Reingresos', ax=ax_reg, color="green", label="Datos Históricos Reales")
                sns.lineplot(x=T_future['Time'], y=preds, ax=ax_reg, color="orange", marker="X", markersize=10, linestyle="--", label="Predicción ML")
                ax_reg.set_xticks(range(1, len(tendencia) + len(per_futuros) + 1))
                ax_reg.set_xticklabels(list(tendencia['PeriodoAcadémico']) + per_futuros, rotation=45)
                ax_reg.set_title("Regresión Lineal: Predicción de Retorno Orgánico", fontweight='bold')
                ax_reg.legend()
                st.pyplot(figE)
            else:
                st.warning("Se requieren al menos 3 periodos históricos con reingresos para trazar la línea de regresión.")

    # =============================================================================
    # PESTAÑA 4: DOCUMENTACIÓN Y EXPORTACIÓN
    # =============================================================================
    with tab4:
        st.header("Guía Metodológica del Sistema")
        st.markdown("""
        ### Mejoras en esta versión:
        1. **Filtros Globales Heredados:** Ahora, si seleccionas una Facultad o Estrato en el panel izquierdo, **todo el simulador y los modelos de Machine Learning se recalcularán** basados exclusivamente en ese perfil poblacional.
        2. **Estándar Visual (Seaborn):** Se migró a la librería académica Seaborn. Sus paletas de colores neutros (`whitegrid`, `Set2`, `viridis`) evitan la fatiga visual, comunican seriedad directiva y eliminan el aspecto "infantil" o excesivamente colorido.
        3. **Nuevos Modelos (Árboles de Decisión):** Además de predecir la cantidad futura, la Pestaña 3 ahora incluye un *DecisionTreeClassifier* de Scikit-Learn que evalúa matemáticamente qué variables sociodemográficas (Estrato y Nivel) definen el perfil del estudiante que retorna.
        4. **Geolocalización:** Se programó un motor de latitud/longitud en la Pestaña 1 que mapea las ciudades del Valle de Aburrá (Medellín, Envigado, Caldas, etc.) basándose en la columna `CIUDADRESIDENCIA`.
        """)
        
        # Botón de exportación unificado
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            candidatos_base.to_excel(writer, sheet_name='Base_Filtrada_A_Contactar', index=False)
            df_proy_sim.to_excel(writer, sheet_name='Proyeccion_Escenario', index=False)
        output.seek(0)
        st.download_button(label="📥 Exportar Proyección y Target para Power BI (.xlsx)",
                           data=output, file_name="Modelo_Reingresos_Unilasallista.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                           
except Exception as e:
    st.error("Error al compilar el modelo de datos. Asegúrate de tener los archivos correctos.")
    st.exception(e)
