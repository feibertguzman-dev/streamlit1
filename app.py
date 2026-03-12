import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Intentar importar Prophet para el Módulo Híbrido
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
    # PANTALLA EMERGENTE DE BIENVENIDA (SOLUCIÓN NATIVA SIN BUGS HTML)
    # =============================================================================
    if not st.session_state['app_iniciada']:
        
        # Inyección de CSS seguro para fuentes y colores
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,600;0,800;1,700&display=swap');
        h1, h2, h3, h4, p, div { font-family: 'Montserrat', sans-serif !important; }
        .b-title { color: #0a2647; font-size: 38px; font-weight: 800; text-align: center; text-transform: uppercase; margin-bottom: 5px;}
        .b-subtitle { color: #4a52c7; font-size: 18px; font-weight: 700; text-align: center; letter-spacing: 1px; margin-bottom: 20px;}
        .b-quote { color: #0a2647; font-size: 24px; font-weight: 700; font-style: italic; text-align: center; margin-bottom: 30px;}
        .b-text { color: #444444; font-size: 18px; text-align: center; max-width: 800px; margin: 0 auto; line-height: 1.6;}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Logos centrados usando columnas nativas
        c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
        try:
            with c2: st.image("logoUnilasalle.png", use_column_width=True)
            with c3: st.image("est.png", use_column_width=True)
        except: pass
        
        # Textos con clases seguras
        st.markdown('<div class="b-title">Inteligencia Analítica y Retención</div>', unsafe_allow_html=True)
        st.markdown('<div class="b-subtitle">VICERRECTORÍA FINANCIERA | CORPORACIÓN UNIVERSITARIA LASALLISTA</div>', unsafe_allow_html=True)
        st.markdown('<div class="b-quote">"Un lugar que te abraza y a la vez te impulsa."</div>', unsafe_allow_html=True)
        st.markdown('<div class="b-text">Bienvenido a la plataforma de Inteligencia de Negocios enfocada en la viabilidad financiera y la retención académica. Este software procesa miles de transacciones históricas para transformarlas en un <b>embudo de recuperación comercial</b>, proyectando el comportamiento orgánico y perfilando el riesgo de deserción.</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Botón de ingreso centrado
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
        
        # ---> NUEVO FILTRO: Periodo Académico <---
        per_sel = st.sidebar.selectbox("Periodo Académico", ["Todos"] + list(sorted(df_crudo['PeriodoAcadémico'].dropna().unique(), reverse=True)))
        
        est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df_crudo['ESTRATO'].dropna().unique())))
        cohorte_sel = st.sidebar.selectbox("Cohorte (Año de Ingreso)", ["Todos"] + list(sorted(df_crudo['AÑOCOHORTE'].dropna().unique(), reverse=True)))
        gen_sel = st.sidebar.selectbox("Género", ["Todos"] + list(sorted(df_crudo['GENERO'].dropna().unique())))

        # Aplicar Filtros a df_base
        df_base = df_crudo.copy()
        if busqueda_txt:
            mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
            df_base = df_base[mask]
        if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
        if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
        if per_sel != "Todos": df_base = df_base[df_base['PeriodoAcadémico'] == per_sel] # <--- Aplicación del filtro
        if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
        if cohorte_sel != "Todos": df_base = df_base[df_base['AÑOCOHORTE'] == cohorte_sel]
        if gen_sel != "Todos": df_base = df_base[df_base['GENERO'] == gen_sel]

        # -----------------------------------------------------------------------------
        # ETL: UNICIDAD CRONOLÓGICA (Identificar Estudiantes Únicos y Activos)
        # -----------------------------------------------------------------------------
        df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
        def clasificar_target(estados):
            lista = estados.tolist()
            if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
            elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
            return 'No Aplica'
            
        estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
        
        # df_univ contiene UNA SOLA FILA por estudiante con su estado terminal absoluto
        df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
        
        # Prospectos Oro: Nivel 5+, inactivos sin reingreso posterior
        df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]
        
        # KPI Cálculo de Activos Reales
        estudiantes_activos = len(df_univ[df_univ['ESTADO'] == 'Estudiante Matriculado'])

        # -----------------------------------------------------------------------------
        # PESTAÑAS DEL SISTEMA (Reordenadas por valor jerárquico)
        # -----------------------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Radiografía Descriptiva", 
            "📞 Gestión Operativa", 
            "⚙️ Simulador Comercial", 
            "🧠 Predicciones Básicas", 
            "🔗 Módulo Híbrido Avanzado",
            "📖 Documentación"
        ])
        
        # =============================================================================
        # 1. ANÁLISIS DESCRIPTIVO (KPIs DE ALTO VALOR)
        # =============================================================================
        with tab1:
            st.header("Radiografía de Tendencias y Población")
            
            # Tarjetas de Alto Valor Analítico
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🟢 Estudiantes Activos (Estado Terminal)", f"{estudiantes_activos:,}")
            k2.metric("📋 Transacciones Históricas Totales", f"{len(df_base):,}")
            k3.metric("🎯 Prospectos de Valor (Retirados Nivel 5+)", f"{len(df_candidatos_finales):,}")
            k4.metric("📈 Alumnos Nuevos (Histórico)", f"{len(df_base[df_base['¿ESNUEVO'] == 'NUEVO']):,}")
            
            st.markdown("---")
            st.markdown("### 1. Evolución Histórica de Ingresos (Nuevos vs Continuidad)")
            df_trend = df_base.groupby(['PeriodoAcadémico', '¿ESNUEVO']).size().reset_index(name='Cantidad')
            fig_lin = px.line(df_trend, x='PeriodoAcadémico', y='Cantidad', color='¿ESNUEVO', markers=True, title="Tendencia Longitudinal de Matrícula", template="plotly_white", color_discrete_sequence=['#0a2647', '#ffcb05'])
            st.plotly_chart(fig_lin, use_container_width=True)
            with st.expander("💡 ¿Cómo leer este gráfico longitudinal?"):
                st.write("Observa las tendencias en el tiempo. Compara la línea azul (Antiguos) con la amarilla (Nuevos). Si la amarilla decae constantemente, la universidad enfrenta un problema de captación; si la azul decae, el problema es de retención de clientes.")
            
            st.markdown("---")
            st.markdown("### 2. Comportamiento Académico y Fricción")
            colA, colB = st.columns(2)
            with colA:
                heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen', color_continuous_scale='Blues', text_auto=True, title="Mapa de Calor: Zonas de Fricción Financiera", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
                with st.expander("💡 ¿Cómo interpretar el Mapa de Calor?"):
                    st.write("Busca el estado 'Retirado' en el eje vertical (Y) y observa en qué columnas o niveles (X) se concentra el color azul oscuro. Esas zonas oscuras son los semestres 'colador' donde la universidad pierde más dinero.")

            with colB:
                fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='GENERO', ax=ax_box, palette="Set2")
                ax_box.set_title("Distribución de Niveles por Estrato y Género", fontweight='bold', color='#0a2647')
                st.pyplot(fig_box)
                with st.expander("💡 ¿Cómo interpretar el Diagrama de Cajas?"):
                    st.write("Este gráfico agrupa a la población. Si las cajas de los estratos 1 y 2 están gráficamente más abajo que las del estrato 5, significa estadísticamente que las poblaciones vulnerables desertan muchos semestres antes.")

            st.markdown("---")
            st.markdown("### 3. Perfilamiento Radial y Mapa Territorial")
            colC, colD = st.columns(2)
            with colC:
                df_radar = df_base.groupby('ESTRATO').size().reset_index(name='Cantidad')
                if not df_radar.empty:
                    fig_radar = go.Figure(data=go.Scatterpolar(r=df_radar['Cantidad'], theta=df_radar['ESTRATO'], fill='toself', name='Población', marker_color='#4a52c7'))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar de Concentración Socioeconómica")
                    st.plotly_chart(fig_radar, use_container_width=True)
                    with st.expander("💡 ¿Cómo leer el Radar?"):
                        st.write("Muestra hacia qué estrato está sesgada la universidad. Si el pico más largo apunta al 'ESTRATO 3', ese es tu nicho (buyer persona) principal.")
            
            with colD:
                coords = { 'MEDELLIN':(6.2442,-75.5812), 'CALDAS':(6.0911,-75.6383), 'ENVIGADO':(6.1759,-75.5917), 'BELLO':(6.3373,-75.5579), 'LA ESTRELLA':(6.1576,-75.6443), 'ITAGUI':(6.1718,-75.6095), 'SABANETA':(6.1515,-75.6166), 'AMAGA':(6.0385,-75.7034), 'COPACABANA':(6.3463,-75.5089), 'BOGOTA':(4.7110,-74.0721) }
                df_geo = df_base.groupby('CIUDADRESIDENCIA').size().reset_index(name='Estudiantes')
                df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
                df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
                df_geo = df_geo.dropna(subset=['Lat'])
                if not df_geo.empty:
                    fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Estudiantes", color="Estudiantes", hover_name="CIUDADRESIDENCIA", color_continuous_scale="Reds", size_max=45, zoom=9, mapbox_style="carto-positron", title="Zonas de Influencia Geográfica")
                    st.plotly_chart(fig_map, use_container_width=True)
                    with st.expander("💡 ¿Cómo leer el Mapa?"):
                        st.write("Burbujas rojas más grandes indican ciudades donde tienes más estudiantes matriculados o inactivos. Ideal para planear campañas de mercadeo tradicional (vallas, volantes).")

        # =============================================================================
        # 2. GESTIÓN OPERATIVA
        # =============================================================================
        with tab2:
            st.header("Directorio de Prospectos (Embudo Depurado)")
            st.info("Esta tabla no contiene registros duplicados ni estudiantes activos. Contiene exclusivamente prospectos viables que cumplen con tu regla de negocio (Nivel 5+, Retirado, no reingresó después).")
            
            with st.expander("💡 Instrucciones de Operación de la Tabla"):
                st.write("""
                1. Utiliza los filtros del panel izquierdo (Ej. Facultad de Ingeniería, Estrato 2).
                2. La tabla inferior se actualizará mostrando SOLO los estudiantes que cumplen esas condiciones.
                3. Haz clic en el botón 'Exportar Base' para descargar el archivo CSV e importarlo a tu sistema de Call Center.
                """)
                
            cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'GENERO', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
            st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=450)
            
            st.download_button(label="📥 Exportar Base para Call Center (.CSV)", data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'), file_name="Leads_Depurados_Unilasallista.csv", mime="text/csv")

        # =============================================================================
        # 3. SIMULADOR COMERCIAL (CON TARJETAS DINÁMICAS)
        # =============================================================================
        with tab3:
            st.header("Motor de Simulación y Retorno Financiero")
            st.markdown("Evalúa qué pasaría financieramente si realizas campañas activas de llamadas a tu base de prospectos actual.")
            
            base_ini = len(df_candidatos_finales)
            
            st.markdown("### 🛠️ Ajuste de Meta Comercial")
            tasa_recup = st.slider("Ajusta el porcentaje (%) de prospectos que crees que aceptarán regresar por ciclo:", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
            
            # Ejecución matemática
            per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
            base_disp = base_ini
            proy_sim = []
            for per in per_futuros:
                reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
                if reing > base_disp: reing = base_disp
                base_disp -= reing
                proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario_Restante': base_disp})
                
            df_proy_sim = pd.DataFrame(proy_sim)
            
            # TARJETAS DINÁMICAS (KPIs)
            st.markdown("### 📊 Proyección de Resultados")
            c1, c2, c3 = st.columns(3)
            c1.metric(label="📥 Base Inicial de Leads (Insumo)", value=f"{base_ini} Alumnos")
            c2.metric(label=f"💰 Matrículas Nuevas Aseguradas (Al {int(tasa_recup*100)}%)", value=f"{df_proy_sim['Reingresos'].sum()} Alumnos")
            c3.metric(label="📉 Base Perdida / Agotada al Final", value=f"{base_disp} Alumnos")
            
            # Gráfico del simulador
            fig_sim, ax_b = plt.subplots(figsize=(12, 5))
            ax_l = ax_b.twinx()
            sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#0a2647', label="Ingresos Logrados")
            sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#ffcb05', marker='o', lw=3, label="Base Pendiente")
            ax_b.set_title("Curva de Decaimiento Comercial vs Captación", fontweight='bold')
            ax_b.set_ylabel("Matrículas Recuperadas")
            ax_l.set_ylabel("Directorio Pendiente (Teléfonos)", color='#ffcb05')
            ax_b.tick_params(axis='x', rotation=45)
            lines1, labels1 = ax_b.get_legend_handles_labels()
            lines2, labels2 = ax_l.get_legend_handles_labels()
            ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax_l.get_legend().remove()
            st.pyplot(fig_sim)
            
            with st.expander("💡 ¿Cómo funciona y cómo leer este Simulador?"):
                st.write("""
                **La Matemática:** Opera bajo el principio de Rendimientos Decrecientes. 
                * **Las Barras Azules** son los estudiantes que te pagan matrícula ese semestre gracias a tu gestión comercial.
                * **La Línea Amarilla** es tu "Inventario" (La tabla de la pestaña Gestión Operativa). Como los que ya se matricularon salen de la lista, la línea amarilla baja. Muestra visualmente que si no nutres tu base con nuevos prospectos, tu Call Center se quedará sin gente a la cual llamar a largo plazo.
                """)

        # =============================================================================
        # 4. PREVISIONES IA BÁSICAS
        # =============================================================================
        with tab4:
            st.header("Modelos de Inercia Orgánica (Scikit-Learn)")
            st.markdown("A diferencia del Simulador, aquí la Inteligencia Artificial predice lo que ocurrirá por pura inercia si la universidad **no realiza campañas activas**.")
            
            ia1, ia2 = st.tabs(["📉 1. Reingresos Inerciales", "🌳 2. Perfil de Riesgo (Árbol)"])
            
            with ia1:
                st.markdown("#### Proyección de Retornos Orgánicos")
                tendencia_reing = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                
                if len(tendencia_reing) > 2:
                    tendencia_reing['Time'] = range(1, len(tendencia_reing) + 1)
                    modelo_r = LinearRegression().fit(tendencia_reing[['Time']], tendencia_reing['Cantidad'])
                    T_fut = pd.DataFrame({'Time': range(tendencia_reing['Time'].max() + 1, tendencia_reing['Time'].max() + 1 + len(per_futuros))})
                    preds_r = [max(0, p) for p in modelo_r.predict(T_fut)]
                    
                    st.metric("Estimación de Reingresos Autónomos a 5 años", f"{int(sum(preds_r))}")
                    
                    fig_r, ax_r = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_reing, x='Time', y='Cantidad', ax=ax_r, color="#4ea8dd", label="Historia (Con margen de error)")
                    ax_r.plot(T_fut['Time'], preds_r, color="#4a52c7", marker="X", linestyle="--", lw=2, label="IA Futura Orgánica")
                    ax_r.set_xticks(range(1, len(tendencia_reing) + len(per_futuros) + 1))
                    ax_r.set_xticklabels(list(tendencia_reing['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_r.legend()
                    st.pyplot(fig_r)
                    
                    with st.expander("💡 ¿Cómo leer esta Regresión?"):
                        st.write("La línea morada punteada representa cuántos alumnos volverán por motivación propia basándose en el comportamiento histórico (línea azul). La sombra alrededor de la línea azul es la 'confianza' del modelo respecto a los picos pasados.")
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
                        st.write("La IA intenta encontrar patrones sociodemográficos. Lee la primera caja (la raíz): Si dice 'Estrato <= 3', significa que el algoritmo partió a los estudiantes en 'Estratos Bajos' (izquierda) y 'Estratos Altos' (derecha) para descubrir qué grupo es más propenso a sufrir una 'Deserción Tardía' (Target).")

        # =============================================================================
        # 5. TABLERO DE CONTROL HÍBRIDO (PROPHET + RANDOM FOREST)
        # =============================================================================
        with tab5:
            st.header("🔗 Tablero Híbrido: Proyección de Demanda Estacional")
            st.markdown("Este módulo funcional reemplaza los cuadernos estáticos de código. Combina la IA de **Prophet (Estacionalidad Meta/Facebook)** con **Random Forest (Sklearn)** para predecir matrículas exactas de un programa.")
            
            with st.expander("💡 ¿Cómo usar y leer este Tablero Híbrido?"):
                st.write("""
                1. Selecciona en el panel inferior el Programa específico que deseas evaluar.
                2. Selecciona el año y el semestre (1 o 2) que deseas predecir.
                3. Haz clic en **'Ejecutar Inteligencia Híbrida'**. El sistema entrenará 200 árboles de decisión en tiempo real.
                4. **Lectura:** Obtendrás el R2 (Si es mayor a 0.80, la predicción es altamente confiable) y el gráfico que sitúa tu predicción futura frente al pasado.
                """)
            
            st.markdown("### ⚙️ Parámetros del Motor de ML")
            
            if not PROPHET_AVAILABLE:
                st.error("🚨 **Dependencia Faltante:** Para ejecutar este modelo, necesitas instalar Prophet. Pídele al ingeniero de despliegue que agregue `prophet` en el archivo `requirements.txt` de GitHub.")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                df_ins = df_crudo[df_crudo['¿ESNUEVO'] == 'NUEVO'].groupby(['PROGRAMA', 'AÑO', 'PERIODO']).size().reset_index(name='TOTAL_INSCRITOS')
                
                prog_obj = col_m1.selectbox("Programa a Proyectar:", sorted(df_ins['PROGRAMA'].unique()))
                anio_obj = col_m2.number_input("Año Futuro (Target):", min_value=2026, max_value=2035, value=2026)
                sem_obj = col_m3.selectbox("Semestre Futuro (Target):", [1, 2])
                
                if st.button("🚀 Ejecutar Inteligencia Híbrida (Prophet + RF)", type="primary"):
                    with st.spinner('Entrenando Redes y Series de Tiempo Estacionales...'):
                        
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
                            st.error("Datos históricos insuficientes para entrenar el híbrido. Requiere más volumen histórico.")
                        else:
                            # 2. Pipeline Random Forest
                            X = df_train[["AÑO", "PERIODO", "yhat_prophet"]]
                            X = pd.concat([X, pd.get_dummies(df_train[["PROGRAMA"]], drop_first=True)], axis=1)
                            y = df_train["y"]
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
                            rf_model.fit(X_train, y_train)
                            
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
                                
                                # Tarjetas Dinámicas de Resultados
                                st.markdown("---")
                                st.markdown(f"### 📊 Resultados Proyectados: **{prog_obj}**")
                                res1, res2, res3 = st.columns(3)
                                res1.metric(label=f"🎓 Demanda Estimada ({anio_obj}-{sem_obj})", value=f"{int(round(final_pred))} Alumnos")
                                res2.metric(label="✅ Varianza Explicada (R²)", value=f"{r2_val:.2f}")
                                res3.metric(label="📉 Error Absoluto Medio (MAE)", value=f"{mae_val:.2f}")
                                
                                # Gráfica Comparativa
                                fig_hyb, ax_hyb = plt.subplots(figsize=(12, 5))
                                subset_fut['Periodo_Str'] = subset_fut['AÑO'].astype(str) + '-' + subset_fut['PERIODO'].astype(str)
                                sns.lineplot(data=subset_fut, x='Periodo_Str', y='y', marker='o', ax=ax_hyb, color='#0a2647', label='Histórico Real')
                                ax_hyb.scatter(f"{anio_obj}-{sem_obj}", final_pred, color='#e64787', s=200, marker='X', label=f'Predicción Híbrida: {int(final_pred)}')
                                ax_hyb.set_title(f"Evolución Histórica y Predicción de Mercado", fontweight='bold')
                                ax_hyb.set_ylabel("Inscritos")
                                ax_hyb.set_xlabel("Año - Semestre")
                                ax_hyb.tick_params(axis='x', rotation=45)
                                ax_hyb.legend()
                                st.pyplot(fig_hyb)
                                
                            else:
                                st.error("No hay historial suficiente para este programa específico.")

        # =============================================================================
        # 6. BIBLIOTECA DOCUMENTAL (NIVEL TESIS / PROFUNDO)
        # =============================================================================
        with tab6:
            st.header("📖 Biblioteca de Fundamentación Analítica")
            doc1, doc2, doc3, doc4 = st.tabs(["1. Justificación Estratégica", "2. Unicidad y Motor ETL", "3. Filtros y Variables", "4. Algoritmos de Machine Learning"])
            
            with doc1:
                st.markdown("### 1. Justificación y Fundamentación del Proyecto")
                st.write("La deserción universitaria representa un impacto grave a los flujos de caja y a la rentabilidad de la **Corporación Universitaria Lasallista**. Este proyecto de investigación convierte la postura reactiva frente a la deserción en una **postura predictiva e inteligente**. Al aislar estadísticamente a los estudiantes que cursaron Niveles Superiores (5 en adelante), se obtiene un nicho que posee un 'costo hundido' (tiempo e inversión) muy alto, incrementando drásticamente la rentabilidad de su retorno en gestión comercial.")
            
            with doc2:
                st.markdown("### 2. Principio de Unicidad Cronológica y Resolución de Conflictos (ETL)")
                st.write("El repositorio institucional alberga datos transaccionales, donde un mismo individuo (`DOCUMENTOIDENTIDAD`) genera múltiples registros. El requerimiento establece una regla innegociable: **si un estudiante se había retirado y posteriormente reingresó, el modelo debe consumir su estado más reciente, anulando el retiro histórico.** Para ello, se implementó un proceso ETL que ordena la base de datos cronológicamente y aísla el registro absoluto terminal mediante el comando algorítmico `keep='last'`. Esto convierte el historial longitudinal en una fotografía exacta poblacional (evitando duplicar llamadas a personas que ya están estudiando).")
                
            with doc3:
                st.markdown("### 3. Segmentación Dinámica")
                st.write("La arquitectura del Dashboard opera mediante una cascada de filtros globales. Al interactuar con el panel izquierdo (ej. Facultad o Género), la matriz se recalcula en memoria RAM. Esto provoca que **todas** las pestañas (Mapas, Simuladores, IA y Tablas Operativas) reconstruyan sus ecuaciones matemáticas operando únicamente bajo ese subconjunto poblacional.")

            with doc4:
                st.markdown("### 4. Sustentación Algorítmica (Machine Learning)")
                st.write("""
                * **Modelo Híbrido (Prophet + RandomForest):** La joya de la corona del sistema predictivo. Extrae estacionalidad (Semestre 1 vs Semestre 2) con el modelo Prophet, e inyecta ese cálculo a un ensamble de múltiples Árboles Aleatorios (Random Forest) que reduce el error no lineal.
                * **Regresión Lineal Simple (Sklearn):** Proyecta la inercia del mercado sin intervención. La franja sombreada representa el Intervalo de Confianza, dando validez estadística ante la volatilidad real del mercado.
                * **Árboles de Clasificación (Gini):** Debido al desbalance de clases natural en deserción, la IA utiliza un `DecisionTreeClassifier` para perfilar reglas del riesgo sociodemográfico (ej. cruce de niveles y estrato).
                """)

        # =============================================================================
        # FOOTER INSTITUCIONAL (PROPIEDAD INTELECTUAL)
        # =============================================================================
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #606060; font-size: 15px; padding: 25px 0; background-color: #f8f9fa; border-radius: 10px; font-family: sans-serif;">
                <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br><br>
                Desarrollo e Insumo de Investigación aportado por la <strong>Facultad de Ingeniería</strong> bajo la dirección general de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
                Apoyo Técnico de Integración Algorítmica y BI: <strong>Jonathan Berthen Castro</strong><br><br>
                <i>Este tablero interactivo predictivo y aplicativo soporta tecnológica y científicamente al proyecto adscrito al grupo de investigación <strong>G-3IN</strong>.</i>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada o contacta al administrador del sistema.")
    st.exception(e)

