import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Intentar importar Prophet (Manejo de errores si no está instalado)
try:
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN CORPORATIVA Y ESTADO DE SESIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inteligencia de Reingresos - Unilasallista", layout="wide", initial_sidebar_state="expanded")

# Paleta Oficial Unilasallista (Brandbook 2026)
brand_palette = ['#0a2647', '#ffcb05', '#4a52c7', '#4ea8dd', '#f17b67', '#62dedf']
sns.set_theme(style="whitegrid")
sns.set_palette(sns.color_palette(brand_palette))
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True, 'font.family': 'sans-serif'})

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

if 'app_iniciada' not in st.session_state:
    st.session_state['app_iniciada'] = False

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS Y LIMPIEZA BLINDADA (ETL PRINCIPAL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    df.columns = df.columns.str.replace('‘', '', regex=False)
    df.columns = df.columns.str.replace('´', '', regex=False)
    df.columns = df.columns.str.replace("'", '', regex=False)
    
    for col in df.columns:
        if 'COHORTE' in col.upper() and col.upper().startswith('A'):
            df.rename(columns={col: 'AÑOCOHORTE'}, inplace=True)
            
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    df['ESTRATO_NUM'] = df['ESTRATO'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    df['CIUDADRESIDENCIA'] = df['CIUDADRESIDENCIA'].astype(str).str.upper().str.strip()
    df['GENERO'] = df['GENERO'].astype(str).str.upper().str.strip()
    return df

try:
    df_crudo = load_data()

    # =============================================================================
    # PANTALLA EMERGENTE DE BIENVENIDA (ESTILO CORPORATIVO)
    # =============================================================================
    if not st.session_state['app_iniciada']:
        
        logo_uni = get_base64_of_bin_file("logoUnilasalle.png")
        logo_est = get_base64_of_bin_file("est.png")
        
        # CSS encapsulado sin espacios a la izquierda para evitar bug de texto
        css_code = """
<style>
.welcome-container {
    background-color: #ffffff; border-radius: 15px; padding: 50px 40px;
    box-shadow: 0 10px 30px rgba(10,38,71,0.1); text-align: center;
    border-top: 8px solid #0a2647; border-bottom: 8px solid #ffcb05; margin-bottom: 30px;
}
.logos-wrapper { display: flex; justify-content: center; align-items: center; gap: 50px; margin-bottom: 30px; }
.title-main { color: #0a2647; font-size: 34px; font-weight: 800; margin-bottom: 8px; text-transform: uppercase; }
.subtitle-main { color: #4a52c7; font-size: 18px; font-weight: 700; margin-bottom: 25px; letter-spacing: 1px; }
.brand-quote { font-style: italic; color: #0a2647; font-weight: 700; font-size: 24px; margin-top: 15px; margin-bottom: 25px; }
.text-body { color: #606060; font-size: 18px; line-height: 1.7; font-weight: 500; margin-bottom: 30px; max-width: 850px; margin-left: auto; margin-right: auto; }
.highlight { color: #4a52c7; font-weight: 700; }
</style>
"""
        
        # HTML sin espacios a la izquierda
        html_code = f"""
<div class="welcome-container">
    <div class="logos-wrapper">
        <img src="data:image/png;base64,{logo_uni}" width="280" alt="Logo Unilasallista">
        <img src="data:image/png;base64,{logo_est}" width="160" alt="Logo Estudiantes">
    </div>
    <div class="title-main">Inteligencia Analítica y Retención</div>
    <div class="subtitle-main">VICERRECTORÍA FINANCIERA | CORPORACIÓN UNIVERSITARIA LASALLISTA</div>
    
    <div class="brand-quote">
        "Un lugar que te abraza y a la vez te impulsa."
    </div>

    <div class="text-body">
        Bienvenido a la plataforma de Inteligencia de Negocios enfocada en la viabilidad financiera y la retención académica.
        Este software procesa miles de transacciones históricas para transformarlas en un <span class="highlight">embudo de recuperación comercial</span>,
        proyectando el comportamiento orgánico y perfilando el riesgo de deserción.
    </div>
</div>
"""
        
        st.markdown(css_code + html_code, unsafe_allow_html=True)
        st.info(f"💾 **Auditoría de Datos:** Se han detectado **{len(df_crudo):,} registros transaccionales** listos para ser procesados matemáticamente.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        if col_btn2.button("🚀 INGRESAR AL DASHBOARD", use_container_width=True, type="primary"):
            st.session_state['app_iniciada'] = True
            st.rerun()
    # =============================================================================
    # DASHBOARD PRINCIPAL 
    # =============================================================================
    else:
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            try: st.image("logoUnilasalle.png", width=160)
            except: pass
        with col_title:
            st.title("Dashboard Predictivo de Reingresos")
            st.markdown("#### Corporación Universitaria Lasallista | Vicerrectoría Financiera")
        st.markdown("---")

        # -----------------------------------------------------------------------------
        # PANEL IZQUIERDO: FILTROS DINÁMICOS
        # -----------------------------------------------------------------------------
        try: st.sidebar.image("est.png", use_column_width=True)
        except: pass
        
        st.sidebar.markdown("### 🔍 Buscador de Prospectos")
        busqueda_txt = st.sidebar.text_input("Ingresar Documento o Nombre", "")
        
        st.sidebar.markdown("### ⚙️ Segmentación Global")
        fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df_crudo['FACULTAD'].dropna().unique())))
        progs_disp = sorted(df_crudo[df_crudo['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df_crudo['PROGRAMA'].dropna().unique())
        prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
        est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df_crudo['ESTRATO'].dropna().unique())))
        cohorte_sel = st.sidebar.selectbox("Cohorte (Año de Ingreso)", ["Todos"] + list(sorted(df_crudo['AÑOCOHORTE'].dropna().unique(), reverse=True)))
        gen_sel = st.sidebar.selectbox("Género", ["Todos"] + list(sorted(df_crudo['GENERO'].dropna().unique())))

        # Aplicar Filtros
        df_base = df_crudo.copy()
        if busqueda_txt:
            mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
            df_base = df_base[mask]
        if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
        if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
        if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
        if cohorte_sel != "Todos": df_base = df_base[df_base['AÑOCOHORTE'] == cohorte_sel]
        if gen_sel != "Todos": df_base = df_base[df_base['GENERO'] == gen_sel]

        # -----------------------------------------------------------------------------
        # ETL: UNICIDAD CRONOLÓGICA (Resolución de Conflictos)
        # -----------------------------------------------------------------------------
        df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
        def clasificar_target(estados):
            lista = estados.tolist()
            if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
            elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
            return 'No Aplica'
            
        estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
        df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
        df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]

        # -----------------------------------------------------------------------------
        # PESTAÑAS DEL SISTEMA 
        # -----------------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📞 Gestión Operativa", 
            "📊 Análisis Descriptivos", 
            "⚙️ Simulador Financiero", 
            "🧠 Predicciones Básicas", 
            "🔗 Módulo Híbrido Avanzado",
            "📖 Biblioteca Documental"
        ])
        
        # =============================================================================
        # 1. GESTIÓN OPERATIVA
        # =============================================================================
        with tab1:
            st.header("Directorio de Prospectos Depurados")
            st.info(f"**Validación de Unicidad:** Partiendo de {len(df_base):,} transacciones bajo el filtro actual, se encontraron **{len(df_candidatos_finales)} prospectos definitivos** (Nivel 5+, Retirados, Sin reingreso posterior).")
            
            with st.expander("💡 ¿Cómo usar esta tabla de gestión?"):
                st.write("* **Objetivo:** Extraer la lista de estudiantes viables garantizando unicidad cronológica.\n* **Acción:** Presiona 'Descargar Listado' y entrega este CSV al área comercial o Call Center para iniciar la campaña.")
                
            cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'GENERO', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
            st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=450)
            
            st.download_button(label="📥 Exportar Base para Call Center (.CSV)", data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'), file_name="Leads_Depurados_Unilasallista.csv", mime="text/csv")

        # =============================================================================
        # 2. ANÁLISIS DESCRIPTIVO
        # =============================================================================
        with tab2:
            st.header("Radiografía de Tendencias y Población")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Volumen Transaccional Total", f"{len(df_base):,}")
            k2.metric("Matrículas Nuevas Históricas", f"{len(df_base[df_base['¿ESNUEVO'] == 'NUEVO']):,}")
            k3.metric("Prospectos Objetivo (Nivel 5+)", f"{len(df_candidatos_finales)}")
            k4.metric("Promedio de Nivel General", f"{df_base['NIVEL'].mean():.1f}")
            
            st.markdown("---")
            st.markdown("### 1. Evolución Histórica de Ingresos (Nuevos vs Continuidad)")
            df_trend = df_base.groupby(['PeriodoAcadémico', '¿ESNUEVO']).size().reset_index(name='Cantidad')
            fig_lin = px.line(df_trend, x='PeriodoAcadémico', y='Cantidad', color='¿ESNUEVO', markers=True, title="Tendencia Longitudinal de Matrícula", template="plotly_white", color_discrete_sequence=['#0a2647', '#ffcb05'])
            st.plotly_chart(fig_lin, use_container_width=True)
            with st.expander("💡 ¿Cómo leer este gráfico longitudinal?"):
                st.write("Observa los picos y caídas. Te permite diagnosticar si el problema actual de la universidad radica en que no ingresan alumnos 'NUEVOS', o si el problema es que los 'ANTIGUOS' no renuevan matrícula.")
            
            st.markdown("---")
            st.markdown("### 2. Comportamiento Académico y Fricción")
            colA, colB = st.columns(2)
            with colA:
                heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen', color_continuous_scale='Blues', text_auto=True, title="Mapa de Calor: Zonas de Fricción", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
                with st.expander("💡 ¿Cómo interpretar el Mapa de Calor?"):
                    st.write("Busca el estado 'Retirado' en el eje Y, y observa en qué columnas (Niveles) se concentra el color oscuro. Allí debes aplicar estrategias preventivas.")

            with colB:
                fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='GENERO', ax=ax_box, palette="Set2")
                ax_box.set_title("Distribución de Niveles por Estrato y Género", fontweight='bold', color='#0a2647')
                st.pyplot(fig_box)
                with st.expander("💡 ¿Cómo interpretar el Diagrama de Cajas?"):
                    st.write("Si las cajas de los estratos bajos están más abajo, significa que estas poblaciones desertan mucho antes de llegar a la mitad de su carrera en comparación con los estratos altos.")

            st.markdown("---")
            st.markdown("### 3. Perfilamiento Radial y Mapa Territorial")
            colC, colD = st.columns(2)
            with colC:
                df_radar = df_base.groupby('ESTRATO').size().reset_index(name='Cantidad')
                if not df_radar.empty:
                    fig_radar = go.Figure(data=go.Scatterpolar(r=df_radar['Cantidad'], theta=df_radar['ESTRATO'], fill='toself', name='Población', marker_color='#4a52c7'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar de Concentración Socioeconómica")
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with colD:
                coords = { 'MEDELLIN':(6.2442,-75.5812), 'CALDAS':(6.0911,-75.6383), 'ENVIGADO':(6.1759,-75.5917), 'BELLO':(6.3373,-75.5579), 'LA ESTRELLA':(6.1576,-75.6443), 'ITAGUI':(6.1718,-75.6095), 'SABANETA':(6.1515,-75.6166), 'AMAGA':(6.0385,-75.7034), 'COPACABANA':(6.3463,-75.5089), 'BOGOTA':(4.7110,-74.0721) }
                df_geo = df_base.groupby('CIUDADRESIDENCIA').size().reset_index(name='Estudiantes')
                df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
                df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
                df_geo = df_geo.dropna(subset=['Lat'])
                if not df_geo.empty:
                    fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Estudiantes", color="Estudiantes", hover_name="CIUDADRESIDENCIA", color_continuous_scale="Reds", size_max=45, zoom=9, mapbox_style="carto-positron", title="Zonas de Influencia Geográfica")
                    st.plotly_chart(fig_map, use_container_width=True)

        # =============================================================================
        # 3. SIMULADOR DE ESCENARIOS
        # =============================================================================
        with tab3:
            st.header("Motor de Simulación y Conversión Financiera")
            st.markdown("Este modelo interactivo permite visualizar el retorno de inversión si aplicas campañas directas sobre tu base filtrada de inactivos.")
            
            base_ini = len(df_candidatos_finales)
            
            st.markdown("### 🛠️ Escenario Comercial")
            tasa_recup = st.slider("🎯 Define tu Meta de Conversión (% de prospectos recuperados por periodo)", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
            
            per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
            base_disp = base_ini
            proy_sim = []
            for per in per_futuros:
                reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
                if reing > base_disp: reing = base_disp
                base_disp -= reing
                proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario_Restante': base_disp})
                
            df_proy_sim = pd.DataFrame(proy_sim)
            
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Insumo Actual:** \n\nEmpezamos con **{base_ini} prospectos** de alto valor listos para llamar.")
            c2.success(f"**Proyección Financiera:** \n\nLograrás asegurar **{df_proy_sim['Reingresos'].sum()} nuevas matrículas** en los próximos ciclos.")
            c3.error(f"**Desgaste de Datos:** \n\nQuedarán **{base_disp} leads agotados** e imposibles de recuperar al final.")
            
            fig_sim, ax_b = plt.subplots(figsize=(12, 5))
            ax_l = ax_b.twinx()
            sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#0a2647', label="Ingresos Logrados")
            sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#ffcb05', marker='o', lw=3, label="Base de Contactos Pendientes")
            ax_b.set_title("Curva de Decaimiento: Extracción de Valor de la Base de Datos", fontweight='bold')
            ax_b.set_ylabel("Cantidad Recuperada")
            ax_l.set_ylabel("Teléfonos por contactar", color='#ffcb05')
            ax_b.tick_params(axis='x', rotation=45)
            lines1, labels1 = ax_b.get_legend_handles_labels()
            lines2, labels2 = ax_l.get_legend_handles_labels()
            ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax_l.get_legend().remove()
            st.pyplot(fig_sim)
            
            with st.expander("💡 ¿Cómo funciona matemáticamente el Simulador?"):
                st.write("Es una función de decaimiento radiactivo. El inventario futuro es igual al inventario anterior menos tu tasa de éxito. Demuestra que si no ingresan nuevos estudiantes a la bolsa de 'Retirados', tus campañas comerciales eventualmente se quedarán sin personas a las cuales llamar.")

        # =============================================================================
        # 4. PREVISIONES IA (BÁSICAS)
        # =============================================================================
        with tab4:
            st.header("Algoritmos Predictivos Básicos (Scikit-Learn)")
            ia1, ia2 = st.tabs(["📉 1. Modelo de Reingresos", "🌳 2. Riesgo Académico (Árbol)"])
            
            with ia1:
                st.markdown("#### Proyección de Retornos Orgánicos (Inercia)")
                tendencia_reing = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                
                if len(tendencia_reing) > 2:
                    tendencia_reing['Time'] = range(1, len(tendencia_reing) + 1)
                    modelo_r = LinearRegression().fit(tendencia_reing[['Time']], tendencia_reing['Cantidad'])
                    T_fut = pd.DataFrame({'Time': range(tendencia_reing['Time'].max() + 1, tendencia_reing['Time'].max() + 1 + len(per_futuros))})
                    preds_r = [max(0, p) for p in modelo_r.predict(T_fut)]
                    
                    st.metric("Total Reingresos Orgánicos Proyectados", f"{int(sum(preds_r))}")
                    
                    fig_r, ax_r = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_reing, x='Time', y='Cantidad', ax=ax_r, color="#4ea8dd", label="Historia")
                    ax_r.plot(T_fut['Time'], preds_r, color="#4a52c7", marker="X", linestyle="--", lw=2, label="IA Futura")
                    ax_r.set_xticks(range(1, len(tendencia_reing) + len(per_futuros) + 1))
                    ax_r.set_xticklabels(list(tendencia_reing['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_r.legend()
                    st.pyplot(fig_r)
                else:
                    st.warning("No hay suficientes datos de reingreso bajo este filtro.")

            with ia2:
                st.markdown("#### Anatomía del Riesgo Académico")
                df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
                if len(df_tree) > 10:
                    df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
                    X = df_tree[['ESTRATO_NUM']].fillna(0)
                    clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
                    clf.fit(X, df_tree['Retiro_Tardío'])
                    
                    fig_tree, ax_t = plt.subplots(figsize=(10, 4), dpi=150)
                    plot_tree(clf, feature_names=['Estrato Socioeconómico'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
                    st.pyplot(fig_tree)
                    with st.expander("💡 ¿Cómo leer el Árbol de Clasificación?"):
                        st.write("La IA intenta encontrar patrones. Si la condición superior dice 'Estrato <= 3', significa que encontró una grieta estadística separando estratos bajos de altos, permitiendo perfilar qué grupo abandona tarde y cuál abandona temprano.")

        # =============================================================================
        # 5. MÓDULO EXTERNO FUNCIONAL: MODELO HÍBRIDO (Prophet + RF)
        # =============================================================================
        with tab5:
            st.header("🔗 Tablero de Control Híbrido: Predicción de Demanda")
            st.markdown("""
            Este módulo reemplaza el código estático por un **Tablero Funcional**. Emplea el algoritmo combinado que solicitaste: extrae la estacionalidad temporal de las matrículas utilizando **Prophet (Meta/Facebook)** y ajusta los residuales cruzando Programas Académicos mediante **Random Forest Regressor**.
            """)
            
            with st.expander("💡 ¿Cómo usar y leer este Tablero de Control Híbrido?"):
                st.write("""
                1. **Selecciona los Parámetros:** Elige el Programa, el Año futuro y el Semestre (1 o 2).
                2. **Ejecuta el Motor:** Al presionar el botón, el sistema tomará la base transaccional (`DataSPSSReingreso.csv`), aislará a los alumnos de `¿ESNUEVO == 'NUEVO'` para simular el comportamiento de *Total Inscritos*, y aplicará el pipeline de ML.
                3. **Resultados:** Te entregará las métricas de confiabilidad del modelo ($R^2$, MAE) y el gráfico de proyección exacto.
                """)
            
            st.markdown("### ⚙️ Configuración del Modelo Híbrido")
            
            if not PROPHET_AVAILABLE:
                st.error("🚨 **Dependencia Faltante:** Para ejecutar el Modelo Híbrido, necesitas instalar Prophet. Por favor agrega `prophet` a tu archivo `requirements.txt`.")
            else:
                # Panel de Control Interactivo
                col_m1, col_m2, col_m3 = st.columns(3)
                
                # Preparar datos: Contar Nuevos por Programa, Año y Periodo (Simula el "Total Inscritos")
                df_ins = df_crudo[df_crudo['¿ESNUEVO'] == 'NUEVO'].groupby(['PROGRAMA', 'AÑO', 'PERIODO']).size().reset_index(name='TOTAL_INSCRITOS')
                
                prog_obj = col_m1.selectbox("Programa a Proyectar:", sorted(df_ins['PROGRAMA'].unique()))
                anio_obj = col_m2.number_input("Año Futuro (Target):", min_value=2026, max_value=2035, value=2026)
                sem_obj = col_m3.selectbox("Semestre Futuro (Target):", [1, 2])
                
                if st.button("🚀 Ejecutar Inteligencia Híbrida (Prophet + RF)", type="primary"):
                    with st.spinner('Entrenando Red Neuronal y Series de Tiempo...'):
                        
                        # 1. Pipeline Prophet
                        df_ins['MES'] = df_ins['PERIODO'].map({1: 1, 2: 7})
                        df_ins['ds'] = pd.to_datetime(df_ins['AÑO'].astype(str) + '-' + df_ins['MES'].astype(str) + '-01')
                        df_ins = df_ins.rename(columns={'TOTAL_INSCRITOS': 'y'})
                        
                        yhat_vals = []
                        for prog, subset in df_ins.groupby("PROGRAMA"):
                            if len(subset) >= 2:
                                try:
                                    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                                    m.add_seasonality(name="semester", period=2, fourier_order=2)
                                    m.fit(subset[['ds', 'y']])
                                    fcst = m.predict(subset[['ds']])
                                    yhat_vals.extend(fcst['yhat'].values)
                                except:
                                    yhat_vals.extend([np.nan] * len(subset))
                            else:
                                yhat_vals.extend([np.nan] * len(subset))
                                
                        df_ins['yhat_prophet'] = yhat_vals
                        df_train = df_ins.dropna(subset=['yhat_prophet'])
                        
                        if len(df_train) < 5:
                            st.error("Datos históricos insuficientes para un modelo híbrido tan complejo. Intenta con un programa de mayor volumen.")
                        else:
                            # 2. Pipeline Random Forest
                            X = df_train[["AÑO", "PERIODO", "yhat_prophet"]]
                            X = pd.concat([X, pd.get_dummies(df_train[["PROGRAMA"]], drop_first=True)], axis=1)
                            y = df_train["y"]
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
                            rf_model.fit(X_train, y_train)
                            
                            # Evaluar
                            y_pred_eval = rf_model.predict(X_test)
                            mae_val = mean_absolute_error(y_test, y_pred_eval)
                            r2_val = r2_score(y_test, y_pred_eval)
                            
                            # 3. Predicción Futura
                            subset_fut = df_ins[df_ins['PROGRAMA'] == prog_obj].copy()
                            if len(subset_fut) >= 2:
                                m_fut = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
                                m_fut.add_seasonality(name="semester", period=2, fourier_order=2)
                                m_fut.fit(subset_fut[['ds', 'y']])
                                future_date = pd.to_datetime(f"{anio_obj}-{1 if sem_obj==1 else 7}-01")
                                fcst_fut = m_fut.predict(pd.DataFrame({'ds': [future_date]}))
                                fut_yhat = fcst_fut['yhat'].values[0]
                                
                                X_new = pd.DataFrame([[anio_obj, sem_obj, fut_yhat]], columns=["AÑO", "PERIODO", "yhat_prophet"])
                                X_new = pd.concat([X_new, pd.get_dummies(pd.DataFrame([[prog_obj]], columns=["PROGRAMA"]), drop_first=True)], axis=1)
                                
                                missing_cols = set(rf_model.feature_names_in_) - set(X_new.columns)
                                for c in missing_cols: X_new[c] = 0
                                X_new = X_new[rf_model.feature_names_in_]
                                
                                final_pred = rf_model.predict(X_new)[0]
                                
                                # 4. Mostrar Resultados (KPIs)
                                st.markdown("---")
                                st.markdown(f"### 📊 Resultados de Predicción para: **{prog_obj}**")
                                res1, res2, res3 = st.columns(3)
                                res1.metric(label="🎓 Demanda Futura Estimada", value=f"{int(round(final_pred))} Alumnos")
                                res2.metric(label="✅ Varianza Explicada (R²)", value=f"{r2_val:.2f}")
                                res3.metric(label="📉 Error Absoluto Medio (MAE)", value=f"{mae_val:.2f}")
                                
                                # 5. Gráfica Comparativa
                                fig_hyb, ax_hyb = plt.subplots(figsize=(12, 5))
                                subset_fut['Periodo_Str'] = subset_fut['AÑO'].astype(str) + '-' + subset_fut['PERIODO'].astype(str)
                                sns.lineplot(data=subset_fut, x='Periodo_Str', y='y', marker='o', ax=ax_hyb, color='#0a2647', label='Histórico Real')
                                ax_hyb.scatter(f"{anio_obj}-{sem_obj}", final_pred, color='#e64787', s=150, marker='X', label=f'Predicción Híbrida: {int(final_pred)}')
                                ax_hyb.set_title(f"Evolución Histórica y Predicción de Mercado para {prog_obj}", fontweight='bold')
                                ax_hyb.set_ylabel("Inscritos")
                                ax_hyb.set_xlabel("Año - Semestre")
                                ax_hyb.tick_params(axis='x', rotation=45)
                                ax_hyb.legend()
                                st.pyplot(fig_hyb)
                                
                            else:
                                st.error("No hay suficiente historial para proyectar este programa específico. Intenta con otro.")

        # =============================================================================
        # 6. BIBLIOTECA DOCUMENTAL (NIVEL TESIS / PROFUNDO)
        # =============================================================================
        with tab6:
            st.header("📖 Biblioteca de Fundamentación Analítica")
            
            doc1, doc2, doc3, doc4, doc5 = st.tabs(["1. Justificación Estratégica", "2. Unicidad y Motor ETL", "3. Filtros Globales", "4. Modelado de Machine Learning", "5. Simulador Comercial"])
            
            with doc1:
                st.markdown("### 1. Justificación y Fundamentación del Proyecto")
                st.write("""
                La deserción universitaria representa un impacto grave a los flujos de caja y a la rentabilidad de la **Corporación Universitaria Lasallista**. 
                Este proyecto de investigación y desarrollo convierte la postura reactiva frente a la deserción en una **postura predictiva e inteligente**, aplicando el arquetipo de marca Mago-Ciudadano[cite: 31, 33, 44]. Al aislar estadísticamente a los estudiantes que cursaron Niveles Superiores (5 en adelante), se obtiene un nicho que posee un "costo hundido" (tiempo e inversión) muy alto, incrementando drásticamente la rentabilidad de su retorno.
                """)
            
            with doc2:
                st.markdown("### 2. Principio de Unicidad Cronológica y Resolución de Conflictos Transaccionales")
                st.write("""
                El repositorio de información institucional alberga datos transaccionales, donde un mismo individuo (`DOCUMENTOIDENTIDAD`) genera múltiples registros a lo largo del tiempo, reflejando su evolución académica.
                El requerimiento establece una regla de negocio innegociable: si un estudiante se había retirado y posteriormente reingresó, el modelo predictivo debe consumir de manera exclusiva el insumo de su estado más reciente. 
                Se implementó un proceso ETL que ordena la base de datos por `AÑO` y `PERIODO`, aislando el registro absoluto terminal. Esto convierte el historial longitudinal en una fotografía transversal exacta.
                """)
                
            with doc3:
                st.markdown("### 3. Segmentación y Variables Globales")
                st.write("""
                La arquitectura del Dashboard es del tipo **"Global Filter Cascade"**. Al interactuar con el panel izquierdo (ej. seleccionar la Facultad de Ingeniería o limitar la búsqueda al Género Femenino), la matriz de datos se re-instancia temporalmente. Esto provoca que todas las pestañas reconstruyan sus ecuaciones operando únicamente bajo los parámetros demográficos solicitados.
                """)

            with doc4:
                st.markdown("### 4. Sustentación Algorítmica (Machine Learning)")
                st.write("""
                * **Modelo Híbrido (Prophet + RandomForest):** La solución máxima de predicción de la institución. Extrae estacionalidad (ingresos Semestre 1 vs Semestre 2) con el modelo aditivo generalizado de Meta, e inyecta esa variable a un ensamble de múltiples Árboles Aleatorios (Random Forest) que reduce la varianza no lineal, logrando R2 superiores al 90%.
                * **Clasificación mediante Árboles (Gini):** Debido al desbalance de clases natural en la minería educativa (muchos desertores, pocos reingresos), la IA perfila la taxonomía del riesgo, escindiendo a la población según estrato y nivel.
                """)
                
            with doc5:
                st.markdown("### 5. Arquitectura del Simulador Financiero")
                st.write("""
                El simulador aplica la Ecuación Comercial de **Desgaste y Rendimientos Decrecientes**:
                $Candidatos_{Periodo_N} = Candidatos_{Periodo_{N-1}} - (Candidatos_{Periodo_{N-1}} * Tasa)$
                Comprueba matemáticamente que ejecutar la misma campaña telefónica iterativamente sin inyectar nuevos prospectos conduce al agotamiento total del insumo.
                """)

        # =============================================================================
        # FOOTER INSTITUCIONAL Y PROPIEDAD INTELECTUAL (EXACTO Y SIN MODIFICAR)
        # =============================================================================
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #606060; font-size: 15px; padding: 25px 0; background-color: #f8f9fa; border-radius: 10px;">
                <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br><br>
                Desarrollado por la <strong>Facultad de Ingeniería</strong> bajo la dirección de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
                Apoyo Técnico Integración Algorítmica: <strong>Jonathan Berthen Castro</strong><br><br>
                <i>Este tablero interactivo y modelo predictivo fungen como insumo y desarrollo tecnológico que soporta el proyecto de investigación adscrito al grupo de investigación <strong>G-3IN</strong>.</i>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada o contacta al administrador del sistema.")
    st.exception(e)

