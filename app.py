import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -----------------------------------------------------------------------------
# CONFIGURACIÓN CORPORATIVA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sistema de Inteligencia de Reingresos", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 10})

col_logo, col_title = st.columns([1, 4])
with col_logo:
    try: st.image("logoUnilasalle.png", width=180)
    except: st.write("*(Logo Unilasallista)*")
with col_title:
    st.title("Plataforma Analítica y Predictiva de Reingresos")
    st.markdown("#### Vicerrectoría Financiera | Corporación Universitaria Lasallista")
st.markdown("---")

# -----------------------------------------------------------------------------
# CARGA DE DATOS Y LIMPIEZA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    
    # Limpieza blindada de caracteres corruptos en columnas
    df.columns = df.columns.str.replace('‘', '', regex=False)
    df.columns = df.columns.str.replace('´', '', regex=False)
    df.columns = df.columns.str.replace("'", '', regex=False)
    
    for col in df.columns:
        if 'COHORTE' in col.upper() and col.upper().startswith('A'):
            df.rename(columns={col: 'AÑOCOHORTE'}, inplace=True)
            
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    df['ESTRATO_NUM'] = df['ESTRATO'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    df['CIUDADRESIDENCIA'] = df['CIUDADRESIDENCIA'].astype(str).str.upper().str.strip()
    return df

try:
    df = load_data()
    
    # -----------------------------------------------------------------------------
    # PANEL IZQUIERDO: FILTROS INTEGRADOS Y BÚSQUEDA
    # -----------------------------------------------------------------------------
    try: st.sidebar.image("est.png", use_column_width=True)
    except: pass
    
    st.sidebar.markdown("### 🔍 Buscador de Estudiantes")
    busqueda_txt = st.sidebar.text_input("Buscar por Documento o Nombre", "")
    
    st.sidebar.markdown("### ⚙️ Filtros de Segmentación")
    st.sidebar.info("Estos filtros ajustan todas las tablas, proyecciones y mapas simultáneamente.")
    
    fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df['FACULTAD'].dropna().unique())))
    progs_disp = sorted(df[df['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df['PROGRAMA'].dropna().unique())
    prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
    
    est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df['ESTRATO'].dropna().unique())))
    cohorte_sel = st.sidebar.selectbox("Cohorte de Ingreso (Año)", ["Todos"] + list(sorted(df['AÑOCOHORTE'].dropna().unique(), reverse=True)))

    # APLICACIÓN DE FILTROS AL DATAFRAME PRINCIPAL
    df_base = df.copy()
    if busqueda_txt:
        mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | \
               df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
        df_base = df_base[mask]
    if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
    if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
    if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
    if cohorte_sel != "Todos": df_base = df_base[df_base['AÑOCOHORTE'] == cohorte_sel]

    # -----------------------------------------------------------------------------
    # PESTAÑAS DEL SISTEMA (AHORA SON 6)
    # -----------------------------------------------------------------------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Inicio",
        "📞 Gestión y Contacto", 
        "📊 Análisis Descriptivo & Mapa", 
        "⚙️ Simulador de Escenarios", 
        "🧠 Modelos Predictivos (IA)", 
        "📖 Ayuda y Documentación"
    ])
    
    # =============================================================================
    # PROCESO ETL: Principio de Unicidad Cronológica y Resolución de Conflictos
    # =============================================================================
    df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
    
    def clasificar_target(estados):
        lista = estados.tolist()
        if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
        elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
        return 'No Aplica'
        
    estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
    df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
    df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]

    # =============================================================================
    # PESTAÑA 0: INICIO / BIENVENIDA
    # =============================================================================
    with tab0:
        st.markdown("## Bienvenido al Sistema de Inteligencia Analítica de Reingresos")
        st.write("Esta plataforma de Inteligencia de Negocios (BI) y Machine Learning ha sido diseñada para optimizar la toma de decisiones, la planeación presupuestal y la ejecución de campañas comerciales en la **Vicerrectoría Financiera**.")
        
        st.markdown("---")
        # KPIs globales de la base de datos completa (sin filtros) para dar contexto del tamaño del proyecto
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Total Estudiantes Históricos", f"{df['DOCUMENTOIDENTIDAD'].nunique():,}")
        col_k2.metric("Programas Analizados", f"{df['PROGRAMA'].nunique()}")
        col_k3.metric("Facultades Integradas", f"{df['FACULTAD'].nunique()}")
        col_k4.metric("Cohortes Mapeadas", f"{df['AÑOCOHORTE'].nunique()}")
        
        st.markdown("---")
        st.markdown("### 🗺️ Guía de Navegación Rápida")
        st.markdown("""
        Explora las pestañas superiores para acceder a los diferentes módulos del sistema:
        
        * **📞 Gestión y Contacto:** Extrae listados depurados y sin duplicados de los estudiantes viables para llamar hoy mismo.
        * **📊 Análisis Descriptivo & Mapa:** Visualiza la radiografía sociodemográfica y el mapa de calor territorial de nuestra población.
        * **⚙️ Simulador de Escenarios:** Juega con tasas de éxito comercial y proyecta el retorno financiero para los próximos semestres.
        * **🧠 Modelos Predictivos (IA):** Observa qué dice la Inteligencia Artificial sobre la inercia de reingresos y el perfil de deserción.
        * **📖 Ayuda y Documentación:** Consulta la metodología, las reglas de negocio y los procesos de calidad de datos del sistema.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #005A9C;">
            <h4 style="color: #005A9C; margin-top: 0;">Respaldo Tecnológico e Investigativo</h4>
            <p style="font-size: 15px; color: #333;">Este tablero interactivo es un desarrollo tecnológico avanzado creado en la <strong>Facultad de Ingeniería</strong> bajo la dirección general de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
            El sistema funge como insumo y soporte directo al proyecto de investigación adscrito al grupo <strong>G-3IN</strong>, integrando técnicas de minería de datos, modelos estadísticos y gamificación para el análisis empresarial.</p>
        </div>
        """, unsafe_allow_html=True)

    # =============================================================================
    # PESTAÑA 1: TABLA OPERATIVA DE CONTACTO
    # =============================================================================
    with tab1:
        st.header("Directorio Operativo de Contacto")
        st.success(f"📌 **Filtro Activo:** Mostrando base de datos para **{df_base['DOCUMENTOIDENTIDAD'].nunique():,}** estudiantes (Según panel izquierdo).")
        st.markdown(f"**Estudiantes Objetivo:** {len(df_candidatos_finales)} prospectos decantados (Retirados/Aplazados desde Nivel 5 listos para llamar).")
        
        with st.expander("💡 ¿Cómo usar esta tabla de gestión?"):
            st.write("""
            * **Objetivo:** Extraer la lista de estudiantes viables para reingreso garantizando unicidad cronológica.
            * **Funcionamiento:** Esta tabla responde a los filtros de la izquierda. Si filtras por Estrato 3 o por el programa Zootecnia, esta tabla solo mostrará esos perfiles.
            * **Acción:** Haz clic en el botón de abajo para descargar el archivo CSV unificado para el Call Center.
            """)
            
        cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
        df_mostrar = df_candidatos_finales[cols_gestion].copy()
        
        st.dataframe(df_mostrar, use_container_width=True, height=400)
        
        st.download_button(
            label="📥 Descargar Listado Depurado (.CSV)",
            data=df_mostrar.to_csv(index=False, sep=";").encode('utf-8-sig'),
            file_name="Listado_Contacto_Reingresos_Unilasallista.csv",
            mime="text/csv"
        )

    # =============================================================================
    # PESTAÑA 2: ANÁLISIS DESCRIPTIVO Y MAPA
    # =============================================================================
    with tab2:
        st.header("Radiografía Poblacional")
        colA, colB = st.columns(2)
        with colA:
            fig_box, ax_box = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df_base, x='ESTADO', y='NIVEL', hue='GENERO', ax=ax_box, palette="pastel")
            ax_box.set_title("Concentración de Retiros por Nivel", fontweight='bold')
            ax_box.tick_params(axis='x', rotation=45)
            ax_box.set_xlabel("")
            st.pyplot(fig_box)

        with colB:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df_base, y='ESTRATO', hue='ESTADO', ax=ax_hist, palette="deep")
            ax_hist.set_title("Volumen Sociodemográfico", fontweight='bold')
            ax_hist.set_ylabel("")
            st.pyplot(fig_hist)

        st.markdown("---")
        st.markdown("### 📍 Mapa de Calor Territorial")
        
        coords = {
            'MEDELLIN':(6.2442,-75.5812), 'CALDAS':(6.0911,-75.6383), 'ENVIGADO':(6.1759,-75.5917),
            'BELLO':(6.3373,-75.5579), 'LA ESTRELLA':(6.1576,-75.6443), 'ITAGUI':(6.1718,-75.6095),
            'SABANETA':(6.1515,-75.6166), 'AMAGA':(6.0385,-75.7034), 'COPACABANA':(6.3463,-75.5089),
            'SANTA BARBARA':(5.8741,-75.5668), 'FREDONIA':(5.9261,-75.6749), 'RIONEGRO':(6.1551,-75.3737),
            'GIRARDOTA':(6.3768,-75.4457), 'SAN PEDRO':(6.4632,-75.5544), 'BARBOSA':(6.4402,-75.3288),
            'ANGELOPOLIS':(6.0278,-75.7118), 'ANDES':(5.6565,-75.8778), 'CIUDAD BOLIVAR':(5.8491,-76.0152),
            'VENECIA':(5.9613,-75.7369), 'BOGOTA':(4.7110,-74.0721)
        }
        
        df_geo = df_base.groupby('CIUDADRESIDENCIA').size().reset_index(name='Estudiantes')
        df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
        df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
        df_geo = df_geo.dropna(subset=['Lat'])
        
        if not df_geo.empty:
            fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Estudiantes", color="Estudiantes",
                                        hover_name="CIUDADRESIDENCIA", color_continuous_scale="Reds",
                                        size_max=45, zoom=8, mapbox_style="carto-positron")
            st.plotly_chart(fig_map, use_container_width=True)

    # =============================================================================
    # PESTAÑA 3: SIMULADOR DE ESCENARIOS
    # =============================================================================
    with tab3:
        st.header("Simulador de Retorno Financiero (Estrategia Activa)")
        base_ini = len(df_candidatos_finales)
        
        st.markdown(f"El simulador parte de **{base_ini} estudiantes decantados**. Establece tu meta de conversión:")
        tasa_recup = st.slider("🎯 Tasa de Éxito Comercial (% de reingresos logrados por periodo)", min_value=1.0, max_value=50.0, value=10.0, step=1.0) / 100.0
        
        per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
        base_disp = base_ini
        proy_sim = []
        
        for per in per_futuros:
            reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
            if reing > base_disp: reing = base_disp
            base_disp -= reing
            proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario_Restante': base_disp})
            
        df_proy_sim = pd.DataFrame(proy_sim)
        
        fig_sim, ax_b = plt.subplots(figsize=(10, 4))
        ax_l = ax_b.twinx()
        
        sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#4C72B0', label="Nuevos Matriculados")
        sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#C44E52', marker='o', lw=2.5, label="Candidatos sin contactar")
        
        ax_b.set_title("Curva Financiera: Captación vs Agotamiento de Leads", fontweight='bold')
        ax_b.set_ylabel("Cant. Retornados")
        ax_l.set_ylabel("Candidatos Restantes", color='#C44E52')
        ax_b.tick_params(axis='x', rotation=45)
        
        lines1, labels1 = ax_b.get_legend_handles_labels()
        lines2, labels2 = ax_l.get_legend_handles_labels()
        ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax_l.get_legend().remove()
        st.pyplot(fig_sim)

    # =============================================================================
    # PESTAÑA 4: MACHINE LEARNING
    # =============================================================================
    with tab4:
        st.header("Modelos Predictivos (Machine Learning)")
        
        st.subheader("1. Predicción Orgánica: Regresión Lineal")
        tendencia = df_univ[df_univ['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos')
        
        if len(tendencia) > 2:
            tendencia['Time'] = range(1, len(tendencia) + 1)
            X_train, y_train = tendencia[['Time']], tendencia['Reingresos']
            modelo = LinearRegression().fit(X_train, y_train)
            
            T_fut = pd.DataFrame({'Time': range(tendencia['Time'].max() + 1, tendencia['Time'].max() + 1 + len(per_futuros))})
            preds = [max(0, p) for p in modelo.predict(T_fut)]
            
            fig_reg, ax_reg = plt.subplots(figsize=(10, 4))
            sns.regplot(data=tendencia, x='Time', y='Reingresos', ax=ax_reg, color="#2CA02C", label="Histórico (Con Sombra de Error)")
            ax_reg.plot(T_fut['Time'], preds, color="#FF7F0E", marker="X", linestyle="--", label="Predicción Futura ML")
            
            ax_reg.set_xticks(range(1, len(tendencia) + len(per_futuros) + 1))
            ax_reg.set_xticklabels(list(tendencia['PeriodoAcadémico']) + per_futuros, rotation=45)
            ax_reg.legend()
            st.pyplot(fig_reg)
        else:
            st.warning("No hay suficientes semestres con historial de reingresos bajo estos filtros para trazar la línea de regresión.")

        st.markdown("---")
        st.subheader("2. Árbol de Decisión: Perfil de Deserción")
        df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
        if len(df_tree) > 10:
            df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
            X = df_tree[['ESTRATO_NUM']].fillna(0)
            
            clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
            clf.fit(X, df_tree['Retiro_Tardío'])
            
            fig_tree, ax_t = plt.subplots(figsize=(8, 4), dpi=150)
            plot_tree(clf, feature_names=['Estrato'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
            st.pyplot(fig_tree)

    # =============================================================================
    # PESTAÑA 5: DOCUMENTACIÓN EXHAUSTIVA Y DESPLEGABLES
    # =============================================================================
    with tab5:
        st.header("📖 Manual Interactivo y Reglas de Negocio")
        
        with st.expander("⚖️ Principio de Unicidad Cronológica (Resolución de Conflictos ETL)"):
            st.markdown("""
            **El Reto Transaccional:**
            En las bases de datos institucionales, un mismo individuo genera múltiples registros a lo largo del tiempo.
            
            **La Solución Implementada:**
            El sistema ejecuta un procedimiento de Extracción, Transformación y Carga (ETL) en segundo plano que garantiza la integridad transaccional:
            1. **Aislamiento Terminal:** Aísla de manera exclusiva el registro correspondiente a la fecha máxima absoluta (estado terminal).
            2. **Anulación de Eventos Históricos:** Si un estudiante se había retirado, pero posteriormente reingresó, el modelo predictivo asume el reingreso como su estado definitivo y **anula la deserción previa**.
            """)
            
        with st.expander("🎯 Objetivo Estratégico"):
            st.write("Plataforma diseñada para la Vicerrectoría Financiera. Convierte el historial plano de deserciones en un embudo interactivo para planear, presupuestar y proyectar retornos de inversión en campañas de readmisión.")

    # =============================================================================
    # CRÉDITOS Y PROPIEDAD INTELECTUAL (FOOTER GLOBAL)
    # =============================================================================
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666666; font-size: 14px; padding: 20px 0;">
            <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br>
            Desarrollado por la <strong>Facultad de Ingeniería</strong> bajo la dirección de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
            <i>Este tablero interactivo y modelo predictivo fungen como insumo y desarrollo tecnológico que soporta el proyecto de investigación adscrito al grupo de investigación <strong>G-3IN</strong>.</i>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada.")
    st.exception(e)
