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

# Gestión de dependencias para el Modelo Híbrido
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

if 'app_iniciada' not in st.session_state:
    st.session_state['app_iniciada'] = False

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS Y LIMPIEZA BLINDADA (ETL)
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
    # PANTALLA DE BIENVENIDA (PRESENTACIÓN ESTRATÉGICA OPTIMIZADA)
    # =============================================================================
    if not st.session_state['app_iniciada']:
        # Estilos para el texto de bienvenida
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
        .welcome-card {
            font-family: 'Montserrat', sans-serif;
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .main-title { color: #0a2647; font-size: 42px; font-weight: 800; margin-top: 20px; }
        .sub-title { color: #4a52c7; font-size: 18px; font-weight: 600; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 30px; }
        .brand-quote { color: #0a2647; font-size: 28px; font-weight: 400; font-style: italic; margin-bottom: 25px; }
        .hero-text { color: #444444; font-size: 20px; line-height: 1.8; margin-bottom: 40px; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Logo único y centrado usando el comando estándar de Streamlit
        col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
        with col_img2:
            try:
                st.image("logoUnilasalle.png", use_container_width=True)
            except:
                st.warning("Logo no encontrado. Verifique el archivo logoUnilasalle.png")
        
        st.markdown(f"""
        <div class="welcome-card">
            <div class="sub-title">Vicerrectoría Financiera</div>
            <div class="main-title">Estrategia Analítica de Retención</div>
            <div class="brand-quote">"Un lugar que te abraza y a la vez te impulsa"</div>
            <div class="hero-text">
                Bienvenido a la plataforma de <b>Inteligencia de Datos</b> Unilasallista. <br>
                Esta herramienta ha sido diseñada para transformar la información histórica en decisiones estratégicas, 
                permitiéndonos acompañar con precisión el camino de nuestros estudiantes y fortalecer el 
                propósito transformador de nuestra institución.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Botón de acceso
        col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
        if col_b2.button("INGRESAR AL PANEL ESTRATÉGICO", use_container_width=True, type="primary"):
            st.session_state['app_iniciada'] = True
            st.rerun()
            
    # =============================================================================
    # DASHBOARD PRINCIPAL
    # =============================================================================
    else:
        col_logo_dash, col_title_dash = st.columns([1, 4])
        with col_logo_dash:
            try: st.image("logoUnilasalle.png", width=160)
            except: pass
        with col_title_dash:
            st.title("Dashboard Predictivo de Reingresos")
            st.markdown("#### Corporación Universitaria Lasallista | Vicerrectoría Financiera")
        st.markdown("---")

        # PANEL IZQUIERDO: FILTROS
        st.sidebar.markdown("### 🔍 Buscador")
        busqueda_txt = st.sidebar.text_input("Ingresar Documento o Nombre", "")
        
        st.sidebar.markdown("### ⚙️ Segmentación Global")
        fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df_crudo['FACULTAD'].dropna().unique())))
        progs_disp = sorted(df_crudo[df_crudo['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df_crudo['PROGRAMA'].dropna().unique())
        prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
        per_sel = st.sidebar.selectbox("Periodo Académico", ["Todos"] + list(sorted(df_crudo['PeriodoAcadémico'].dropna().unique(), reverse=True)))
        est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df_crudo['ESTRATO'].dropna().unique())))
        cohorte_sel = st.sidebar.selectbox("Cohorte (Año de Ingreso)", ["Todos"] + list(sorted(df_crudo['AÑOCOHORTE'].dropna().unique(), reverse=True)))
        gen_sel = st.sidebar.selectbox("Género", ["Todos"] + list(sorted(df_crudo['GENERO'].dropna().unique())))

        df_base = df_crudo.copy()
        if busqueda_txt:
            mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
            df_base = df_base[mask]
        if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
        if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
        if per_sel != "Todos": df_base = df_base[df_base['PeriodoAcadémico'] == per_sel]
        if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
        if cohorte_sel != "Todos": df_base = df_base[df_base['AÑOCOHORTE'] == cohorte_sel]
        if gen_sel != "Todos": df_base = df_base[df_base['GENERO'] == gen_sel]

        # ETL: UNICIDAD Y RESOLUCIÓN
        df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
        def clasificar_target(estados):
            lista = estados.tolist()
            if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
            elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
            return 'No Aplica'
            
        estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
        df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
        df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]
        estudiantes_activos = len(df_univ[df_univ['ESTADO'] == 'Estudiante Matriculado'])

        # PESTAÑAS
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Radiografía Descriptiva", "📞 Gestión Operativa", "⚙️ Simulador Comercial", 
            "🧠 Predicciones Básicas", "🔗 Módulo Híbrido IA", "📖 Biblioteca Documental"
        ])
        
        # 1. RADIOGRAFÍA
        with tab1:
            st.header("Radiografía de Tendencias y Población")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🟢 Estudiantes Activos (Actual)", f"{estudiantes_activos:,}")
            k2.metric("📋 Transacciones Históricas", f"{len(df_base):,}")
            k3.metric("🎯 Prospectos de Valor (Nivel 5+)", f"{len(df_candidatos_finales):,}")
            k4.metric("📈 Alumnos Nuevos (Histórico)", f"{len(df_base[df_base['¿ESNUEVO'] == 'NUEVO']):,}")
            
            st.markdown("---")
            st.markdown("### 1. Evolución Histórica de Matrícula")
            df_trend = df_base.groupby(['PeriodoAcadémico', '¿ESNUEVO']).size().reset_index(name='Cantidad')
            fig_lin = px.line(df_trend, x='PeriodoAcadémico', y='Cantidad', color='¿ESNUEVO', markers=True, template="plotly_white", color_discrete_sequence=['#0a2647', '#ffcb05'])
            st.plotly_chart(fig_lin, use_container_width=True)
            with st.expander("💡 ¿Cómo leer este gráfico longitudinal?"):
                st.write("Analice la brecha entre nuevos y antiguos para diagnosticar si el reto es la captación o la permanencia.")

            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen', color_continuous_scale='Blues', text_auto=True, title="Mapa de Calor: Zonas de Fricción", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            with colB:
                fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='GENERO', ax=ax_box)
                ax_box.set_title("Distribución Académica por Estrato", fontweight='bold', color='#0a2647')
                st.pyplot(fig_box)

            st.markdown("---")
            st.markdown("### 2. Ubicación Territorial")
            coords = { 'MEDELLIN':(6.2442,-75.5812), 'CALDAS':(6.0911,-75.6383), 'ENVIGADO':(6.1759,-75.5917), 'BELLO':(6.3373,-75.5579), 'LA ESTRELLA':(6.1576,-75.6443), 'ITAGUI':(6.1718,-75.6095), 'SABANETA':(6.1515,-75.6166), 'AMAGA':(6.0385,-75.7034), 'COPACABANA':(6.3463,-75.5089), 'BOGOTA':(4.7110,-74.0721) }
            df_geo = df_base.groupby('CIUDADRESIDENCIA').size().reset_index(name='Estudiantes')
            df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
            df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
            df_geo = df_geo.dropna(subset=['Lat'])
            if not df_geo.empty:
                fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Estudiantes", color="Estudiantes", hover_name="CIUDADRESIDENCIA", color_continuous_scale="Reds", size_max=45, zoom=9, mapbox_style="carto-positron")
                st.plotly_chart(fig_map, use_container_width=True)

        # 2. GESTIÓN OPERATIVA
        with tab2:
            st.header("Directorio Depurado para Contacto")
            st.info(f"Se han identificado **{len(df_candidatos_finales)} prospectos** que cumplen con los criterios de viabilidad.")
            cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'GENERO', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
            st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=450)
            st.download_button(label="📥 Exportar Base para Call Center (.CSV)", data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'), file_name="Prospectos_Unilasallista.csv", mime="text/csv")

        # 3. SIMULADOR
        with tab3:
            st.header("Motor de Proyección Financiera")
            base_ini = len(df_candidatos_finales)
            tasa_recup = st.slider("🎯 Meta de Conversión Comercial (%)", 1.0, 50.0, 15.0, 1.0) / 100.0
            
            per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
            base_disp = base_ini
            proy_sim = []
            for per in per_futuros:
                reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
                if reing > base_disp: reing = base_disp
                base_disp -= reing
                proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario': base_disp})
            df_proy_sim = pd.DataFrame(proy_sim)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("📥 Base Inicial", f"{base_ini} leads")
            c2.metric("💰 Matrículas Estimadas", f"{df_proy_sim['Reingresos'].sum()} alumnos")
            c3.metric("📉 Base Agotada", f"{base_disp} leads")
            
            fig_sim, ax_b = plt.subplots(figsize=(12, 5))
            ax_l = ax_b.twinx()
            sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#0a2647', label="Reingresos")
            sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario', ax=ax_l, color='#ffcb05', marker='o', lw=3, label="Base Pendiente")
            st.pyplot(fig_sim)

        # 4. PREDICCIONES
        with tab4:
            st.header("Modelos de Inercia Orgánica")
            ia1, ia2 = st.tabs(["📉 Reingresos Inerciales", "🌳 Riesgo Académico"])
            with ia1:
                tendencia_reing = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                if len(tendencia_reing) > 2:
                    tendencia_reing['Time'] = range(1, len(tendencia_reing) + 1)
                    modelo_r = LinearRegression().fit(tendencia_reing[['Time']], tendencia_reing['Cantidad'])
                    T_fut = pd.DataFrame({'Time': range(tendencia_reing['Time'].max() + 1, tendencia_reing['Time'].max() + 1 + 12)})
                    preds_r = [max(0, p) for p in modelo_r.predict(T_fut)]
                    st.metric("Total Reingresos Proyectados (IA)", f"{int(sum(preds_r))}")
                    fig_r, ax_r = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_reing, x='Time', y='Cantidad', ax=ax_r, color="#4ea8dd")
                    ax_r.plot(T_fut['Time'], preds_r, color="#4a52c7", marker="X", linestyle="--")
                    st.pyplot(fig_r)

        # 5. HÍBRIDO
        with tab5:
            st.header("Tablero de Control: Predicción de Demanda Estacional")
            if not PROPHET_AVAILABLE:
                st.error("Requiere prophet instalado.")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                df_ins = df_crudo[df_crudo['¿ESNUEVO'] == 'NUEVO'].groupby(['PROGRAMA', 'AÑO', 'PERIODO']).size().reset_index(name='y')
                prog_obj = col_m1.selectbox("Programa:", sorted(df_ins['PROGRAMA'].unique()))
                anio_obj = col_m2.number_input("Año Futuro:", 2026, 2035, 2026)
                sem_obj = col_m3.selectbox("Semestre:", [1, 2])
                if st.button("🚀 Ejecutar IA Híbrida", type="primary"):
                    df_ins['ds'] = pd.to_datetime(df_ins['AÑO'].astype(str) + '-' + df_ins['PERIODO'].map({1: 1, 2: 7}).astype(str) + '-01')
                    sub_fut = df_ins[df_ins['PROGRAMA'] == prog_obj].copy()
                    if len(sub_fut) >= 2:
                        m = Prophet(yearly_seasonality=False).add_seasonality(name="sem", period=2, fourier_order=2).fit(sub_fut[['ds', 'y']])
                        fut_y = m.predict(pd.DataFrame({'ds': [pd.to_datetime(f"{anio_obj}-{1 if sem_obj==1 else 7}-01")]}))['yhat'].values[0]
                        st.metric("Demanda Estimada", f"{int(round(fut_y))} alumnos")
                        fig_h, ax_h = plt.subplots(figsize=(12, 5))
                        sns.lineplot(data=sub_fut, x='ds', y='y', ax=ax_h, color='#0a2647', marker='o')
                        ax_h.scatter(pd.to_datetime(f"{anio_obj}-{1 if sem_obj==1 else 7}-01"), fut_y, color='red', s=100, marker='X')
                        st.pyplot(fig_h)

        # 6. DOCUMENTACIÓN
        with tab6:
            st.header("Fundamentación Analítica")
            d1, d2 = st.tabs(["Metodología", "Algoritmos"])
            with d1:
                st.write("El motor ETL garantiza que si un estudiante reingresó tras un retiro, la deserción se anula del inventario comercial.")
            with d2:
                st.write("Se emplea Prophet para estacionalidad y Random Forest para corrección no lineal.")

        # FOOTER
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #606060; font-size: 15px; padding: 25px 0; background-color: #f8f9fa; border-radius: 10px;">
                <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br><br>
                Desarrollo e Insumo de Investigación aportado por la <strong>Facultad de Ingeniería</strong> bajo la dirección general de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
                Apoyo Técnico de Integración: <strong>Jonathan Berthen Castro</strong><br><br>
                <i>Desarrollo tecnológico que soporta al proyecto de investigación del grupo <strong>G-3IN</strong>.</i>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error en ejecución. Verifique el archivo DataSPSSReingreso.csv")
    st.exception(e)
