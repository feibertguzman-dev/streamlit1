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

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN CORPORATIVA Y ESTADO DE SESIÓN (PANTALLA EMERGENTE)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inteligencia de Reingresos - Unilasallista", layout="wide", initial_sidebar_state="expanded")

# Actualización de paleta Seaborn usando colores del Brandbook Unilasallista
brand_palette = ['#0a2647', '#ffcb05', '#4a52c7', '#4ea8dd', '#f17b67', '#62dedf']
sns.set_theme(style="whitegrid")
sns.set_palette(sns.color_palette(brand_palette))
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True, 'font.family': 'sans-serif'})

# Función para cargar imágenes en Base64 y usarlas en HTML (Centrado perfecto)
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
    df['GENERO'] = df['GENERO'].astype(str).str.upper().str.strip()
    return df

try:
    df_crudo = load_data()

    # =============================================================================
    # PANTALLA EMERGENTE DE BIENVENIDA (IDENTIDAD DE MARCA OFICIAL)
    # =============================================================================
    if not st.session_state['app_iniciada']:
        
        logo_uni = get_base64_of_bin_file("logoUnilasalle.png")
        logo_est = get_base64_of_bin_file("est.png")
        
        # Aplicación estricta del Brandbook 2026 (Montserrat + Colores HEX)
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,400;0,500;0,700;0,800;1,700&display=swap');
        
        .welcome-container {{
            font-family: 'Montserrat', sans-serif;
            background-color: #ffffff;
            border-radius: 15px;
            padding: 50px 40px;
            box-shadow: 0 10px 30px rgba(10,38,71,0.1);
            text-align: center;
            border-top: 8px solid #0a2647; /* Azul Principal */
            border-bottom: 8px solid #ffcb05; /* Amarillo Principal */
            margin-bottom: 30px;
        }}
        .logos-wrapper {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 50px;
            margin-bottom: 30px;
        }}
        .title-main {{
            color: #0a2647; 
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: -0.5px;
        }}
        .subtitle-main {{
            color: #4a52c7; /* Morado Institucional */
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 25px;
            letter-spacing: 1px;
        }}
        .brand-quote {{
            font-style: italic;
            color: #0a2647;
            font-weight: 700;
            font-size: 24px;
            margin-top: 15px;
            margin-bottom: 25px;
        }}
        .text-body {{
            color: #606060; /* Gris de lectura */
            font-size: 18px;
            line-height: 1.7;
            font-weight: 500;
            margin-bottom: 30px;
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
        }}
        .highlight {{
            color: #4a52c7;
            font-weight: 700;
        }}
        </style>
        
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
                proyectando el comportamiento orgánico y perfilando el riesgo de deserción con el propósito de acompañar a cada estudiante en su proceso.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💾 **Estado de la Base de Datos:** Se han detectado **{len(df_crudo):,} registros transaccionales** listos para ser procesados matemáticamente.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        if col_btn2.button("🚀 INGRESAR A LA PLATAFORMA", use_container_width=True, type="primary"):
            st.session_state['app_iniciada'] = True
            st.rerun()
            
    # =============================================================================
    # DASHBOARD PRINCIPAL (DESPUÉS DE INICIAR SESIÓN)
    # =============================================================================
    else:
        # Encabezado del Dashboard
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            try: st.image("logoUnilasalle.png", width=160)
            except: pass
        with col_title:
            st.title("Dashboard Predictivo de Reingresos")
            st.markdown("#### Corporación Universitaria Lasallista | Vicerrectoría Financiera")
        st.markdown("---")

        # -----------------------------------------------------------------------------
        # PANEL IZQUIERDO: FILTROS INTEGRADOS (CON GÉNERO)
        # -----------------------------------------------------------------------------
        try: st.sidebar.image("est.png", use_column_width=True)
        except: pass
        
        st.sidebar.markdown("### 🔍 Buscador de Prospectos")
        busqueda_txt = st.sidebar.text_input("Ingresar Documento o Nombre", "")
        
        st.sidebar.markdown("### ⚙️ Segmentación Dinámica")
        st.sidebar.info("Estos filtros re-entrenan todos los algoritmos y gráficos en tiempo real.")
        
        fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df_crudo['FACULTAD'].dropna().unique())))
        progs_disp = sorted(df_crudo[df_crudo['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df_crudo['PROGRAMA'].dropna().unique())
        prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
        
        est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df_crudo['ESTRATO'].dropna().unique())))
        cohorte_sel = st.sidebar.selectbox("Cohorte (Año de Ingreso)", ["Todos"] + list(sorted(df_crudo['AÑOCOHORTE'].dropna().unique(), reverse=True)))
        
        generos_disp = sorted(df_crudo['GENERO'].dropna().unique())
        gen_sel = st.sidebar.selectbox("Género", ["Todos"] + list(generos_disp))

        # APLICACIÓN DE FILTROS AL DATAFRAME
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
        # RESOLUCIÓN ETL: UNICIDAD CRONOLÓGICA (Target de Gestión)
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
            "🧠 Predicciones IA", 
            "🔗 Módulo Proyecto IA",
            "📖 Biblioteca Documental"
        ])
        
        # =============================================================================
        # 1. GESTIÓN OPERATIVA
        # =============================================================================
        with tab1:
            st.header("Directorio de Prospectos Depurados")
            st.info(f"**Validación de Unicidad:** Partiendo de {len(df_base):,} transacciones bajo el filtro actual, el sistema aisló el historial para no repetir estudiantes. Se encontraron **{len(df_candidatos_finales)} prospectos definitivos** que superaron el Nivel 4 y actualmente están retirados sin haber reingresado posteriormente.")
                
            cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'GENERO', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
            st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=450)
            
            st.download_button(
                label="📥 Exportar Base para Call Center (.CSV)",
                data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'),
                file_name="Leads_Depurados_Unilasallista.csv",
                mime="text/csv"
            )

        # =============================================================================
        # 2. ANÁLISIS DESCRIPTIVO
        # =============================================================================
        with tab2:
            st.header("Radiografía de Tendencias y Población")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Volumen Transaccional Total", f"{len(df_base):,}")
            k2.metric("Matrículas Nuevas Históricas", f"{len(df_base[df_base['¿ESNUEVO'] == 'NUEVO']):,}")
            k3.metric("Prospectos Objetivo (Nivel 5+)", f"{len(df_candidatos_finales)}")
            k4.metric("Promedio de Nivel", f"{df_base['NIVEL'].mean():.1f}")
            
            st.markdown("---")
            st.markdown("### 1. Evolución Histórica de Ingresos (Nuevos vs Continuidad)")
            df_trend = df_base.groupby(['PeriodoAcadémico', '¿ESNUEVO']).size().reset_index(name='Cantidad')
            fig_lin = px.line(df_trend, x='PeriodoAcadémico', y='Cantidad', color='¿ESNUEVO', markers=True,
                              title="Tendencia Longitudinal de Matrícula por Semestre", template="plotly_white",
                              color_discrete_sequence=['#0a2647', '#ffcb05'])
            st.plotly_chart(fig_lin, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 2. Comportamiento Académico y Fricción")
            colA, colB = st.columns(2)
            with colA:
                heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen',
                                              color_continuous_scale='Blues', text_auto=True,
                                              title="Mapa de Calor: Zonas de Fricción Financiera", template="plotly_white")
                fig_heat.update_layout(xaxis_title="Nivel Académico Cursado", yaxis_title="Estado Final")
                st.plotly_chart(fig_heat, use_container_width=True)

            with colB:
                fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='GENERO', ax=ax_box)
                ax_box.set_title("Distribución de Niveles por Estrato y Género", fontweight='bold', color='#0a2647')
                ax_box.legend(loc='lower right', fontsize='small')
                st.pyplot(fig_box)

            st.markdown("---")
            st.markdown("### 3. Perfilamiento Radial y Mapa Territorial")
            colC, colD = st.columns(2)
            
            with colC:
                df_radar = df_base.groupby('ESTRATO').size().reset_index(name='Cantidad')
                if not df_radar.empty:
                    fig_radar = go.Figure(data=go.Scatterpolar(
                      r=df_radar['Cantidad'],
                      theta=df_radar['ESTRATO'],
                      fill='toself',
                      name='Población',
                      marker_color='#4a52c7'
                    ))
                    fig_radar.update_layout(
                      polar=dict(radialaxis=dict(visible=True)),
                      title="Radar de Concentración Socioeconómica",
                      showlegend=False
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            with colD:
                coords = {
                    'MEDELLIN':(6.2442,-75.5812), 'CALDAS':(6.0911,-75.6383), 'ENVIGADO':(6.1759,-75.5917),
                    'BELLO':(6.3373,-75.5579), 'LA ESTRELLA':(6.1576,-75.6443), 'ITAGUI':(6.1718,-75.6095),
                    'SABANETA':(6.1515,-75.6166), 'AMAGA':(6.0385,-75.7034), 'COPACABANA':(6.3463,-75.5089),
                    'BOGOTA':(4.7110,-74.0721)
                }
                df_geo = df_base.groupby('CIUDADRESIDENCIA').size().reset_index(name='Estudiantes')
                df_geo['Lat'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[0])
                df_geo['Lon'] = df_geo['CIUDADRESIDENCIA'].map(lambda x: coords.get(x, (None, None))[1])
                df_geo = df_geo.dropna(subset=['Lat'])
                
                if not df_geo.empty:
                    fig_map = px.scatter_mapbox(df_geo, lat="Lat", lon="Lon", size="Estudiantes", color="Estudiantes",
                                                hover_name="CIUDADRESIDENCIA", color_continuous_scale="Reds",
                                                size_max=45, zoom=9, mapbox_style="carto-positron",
                                                title="Zonas de Influencia Geográfica")
                    st.plotly_chart(fig_map, use_container_width=True)

        # =============================================================================
        # 3. SIMULADOR DE ESCENARIOS
        # =============================================================================
        with tab3:
            st.header("Motor de Simulación y Conversión Financiera")
            st.markdown("Este modelo te permite traducir datos abstractos en estrategias operativas.")
            
            base_ini = len(df_candidatos_finales)
            st.markdown("### 🛠️ Paso 1: Establecer Parámetros")
            tasa_recup = st.slider("🎯 Define tu Meta de Conversión (% de prospectos que aceptarán regresar)", min_value=1.0, max_value=50.0, value=15.0, step=1.0) / 100.0
            
            st.markdown("### 📊 Paso 2: Ejecución Matemática")
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
            c1.info(f"**Insumo:** \n\nEmpezamos con **{base_ini} prospectos** listos para llamar en la base de datos.")
            c2.success(f"**Proyección:** \n\nCon un {int(tasa_recup*100)}% de éxito, asegurarás **{df_proy_sim['Reingresos'].sum()} nuevas matrículas**.")
            c3.error(f"**Desgaste:** \n\nQuedarán **{base_disp} leads imposibles** de recuperar al final de tu ciclo estratégico.")
            
            fig_sim, ax_b = plt.subplots(figsize=(12, 5))
            ax_l = ax_b.twinx()
            sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#0a2647', label="Reingresos (Caja Financiera)")
            sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#ffcb05', marker='o', lw=3, label="Base de Contactos Pendientes")
            
            ax_b.set_title("Curva de Decaimiento: Extracción de Valor de la Base de Datos", fontweight='bold')
            ax_b.set_ylabel("Nuevos Reingresados (Cantidad)")
            ax_l.set_ylabel("Teléfonos por contactar", color='#ffcb05')
            ax_b.tick_params(axis='x', rotation=45)
            
            lines1, labels1 = ax_b.get_legend_handles_labels()
            lines2, labels2 = ax_l.get_legend_handles_labels()
            ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax_l.get_legend().remove()
            st.pyplot(fig_sim)

        # =============================================================================
        # 4. PREVISIONES IA (REINGRESOS Y NUEVOS)
        # =============================================================================
        with tab4:
            st.header("Algoritmos Predictivos de Matriculación (Scikit-Learn)")
            st.markdown("Si la universidad no hace ninguna campaña y se deja llevar por la tendencia orgánica, esto es lo que la Inteligencia Artificial prevé que sucederá:")
            
            ia1, ia2, ia3 = st.tabs(["📉 1. Modelo de Reingresos", "🚀 2. Modelo de NUEVOS ESTUDIANTES", "🌳 3. Riesgo Académico"])
            
            with ia1:
                st.markdown("#### Proyección de Retornos Orgánicos")
                tendencia_reing = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                
                uso_global = False
                if len(tendencia_reing) <= 2:
                    uso_global = True
                    tendencia_reing = df_crudo[df_crudo['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                    st.warning("⚠️ **Alerta:** Los filtros actuales redujeron demasiado el historial. Se muestra la proyección predictiva GLOBAL.")

                if len(tendencia_reing) > 2:
                    tendencia_reing['Time'] = range(1, len(tendencia_reing) + 1)
                    modelo_r = LinearRegression().fit(tendencia_reing[['Time']], tendencia_reing['Cantidad'])
                    T_fut = pd.DataFrame({'Time': range(tendencia_reing['Time'].max() + 1, tendencia_reing['Time'].max() + 1 + len(per_futuros))})
                    preds_r = [max(0, p) for p in modelo_r.predict(T_fut)]
                    
                    fig_r, ax_r = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_reing, x='Time', y='Cantidad', ax=ax_r, color="#4ea8dd", label="Historia (Con margen estadístico)")
                    ax_r.plot(T_fut['Time'], preds_r, color="#4a52c7", marker="X", linestyle="--", lw=2, label="Proyección Futura (IA)")
                    ax_r.set_xticks(range(1, len(tendencia_reing) + len(per_futuros) + 1))
                    ax_r.set_xticklabels(list(tendencia_reing['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_r.legend()
                    st.pyplot(fig_r)
                else:
                    st.error("No hay datos suficientes para proyectar.")

            with ia2:
                st.markdown("#### Proyección de Captación (Matrícula Nueva)")
                tendencia_nuevos = df_base[df_base['¿ESNUEVO'] == 'NUEVO'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                
                if len(tendencia_nuevos) > 2:
                    tendencia_nuevos['Time'] = range(1, len(tendencia_nuevos) + 1)
                    modelo_n = LinearRegression().fit(tendencia_nuevos[['Time']], tendencia_nuevos['Cantidad'])
                    T_fut_n = pd.DataFrame({'Time': range(tendencia_nuevos['Time'].max() + 1, tendencia_nuevos['Time'].max() + 1 + len(per_futuros))})
                    preds_n = [max(0, p) for p in modelo_n.predict(T_fut_n)]
                    
                    k1, k2 = st.columns(2)
                    k1.metric("Nuevos Estudiantes Estimados (Próximos 5 años)", f"{int(sum(preds_n)):,}")
                    k2.success("El algoritmo detecta la inercia del volumen histórico de captación e intuye matemáticamente las futuras cuotas de mercado.")
                    
                    fig_n, ax_n = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_nuevos, x='Time', y='Cantidad', ax=ax_n, color="#0a2647", label="Historia Real")
                    ax_n.plot(T_fut_n['Time'], preds_n, color="#ffcb05", marker="o", linestyle="--", lw=2, label="Previsión Futura (IA)")
                    ax_n.set_xticks(range(1, len(tendencia_nuevos) + len(per_futuros) + 1))
                    ax_n.set_xticklabels(list(tendencia_nuevos['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_n.set_title("Proyección Regresiva: Alumnos Nuevos", fontweight='bold')
                    ax_n.legend()
                    st.pyplot(fig_n)
                else:
                    st.warning("Faltan datos de estudiantes nuevos para este filtro.")

            with ia3:
                st.markdown("#### Anatomía del Desertor (Riesgo Estadístico)")
                df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
                if len(df_tree) > 10:
                    df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
                    X = df_tree[['ESTRATO_NUM']].fillna(0)
                    clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
                    clf.fit(X, df_tree['Retiro_Tardío'])
                    
                    fig_tree, ax_t = plt.subplots(figsize=(10, 4), dpi=150)
                    plot_tree(clf, feature_names=['Estrato Socioeconómico'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
                    st.pyplot(fig_tree)
                else:
                    st.warning("Poca base de datos para generar el árbol de decisión.")

        # =============================================================================
        # 5. INTEGRACIÓN PROYECTO IA (GITHUB EXTERNO)
        # =============================================================================
        with tab5:
            st.header("🔗 Módulo Externo de Integración: Proyecto IA")
            st.markdown("""
            Esta pestaña ha sido diseñada arquitectónicamente como un **puerto de enlace (Iframe / Placeholder)** para albergar el desarrollo independiente publicado en GitHub (`https://github.com/alejoruizr/Proyecto_IA.git`).
            """)
            
            st.info("""
            **Pasos para el Administrador del Sistema:**
            1. Si la aplicación de GitHub está desplegada en la web (ej. Streamlit Cloud), puedes embeberla aquí usando `st.components.v1.iframe("URL", height=800)`.
            2. Alternativamente, puedes clonar las carpetas del repositorio en la misma raíz de este proyecto y ejecutar los modelos de forma nativa.
            """)
            
            st.markdown("""
            <div style="text-align: center; border: 2px dashed #0a2647; border-radius: 10px; padding: 50px; background-color: #f8f9fa;">
                <h3 style="color: #0a2647;">🔌 ESPACIO RESERVADO PARA INTEGRACIÓN GITHUB</h3>
                <p><strong>Repositorio:</strong> <code>alejoruizr/Proyecto_IA</code></p>
                <p><em>(El código o dashboard externo se renderizará automáticamente aquí una vez se configure la URL de despliegue).</em></p>
            </div>
            """, unsafe_allow_html=True)

        # =============================================================================
        # 6. BIBLIOTECA DOCUMENTAL (NIVEL TESIS)
        # =============================================================================
        with tab6:
            st.header("Biblioteca de Metodología Analítica")
            
            doc1, doc2, doc3, doc4, doc5 = st.tabs(["1. Justificación Estratégica", "2. Arquitectura de Datos (ETL)", "3. Ecuación Financiera", "4. Modelado Predictivo (IA)", "5. Glosario de Variables"])
            
            with doc1:
                st.markdown("### 1. Justificación y Fundamentación del Proyecto")
                st.write("La deserción universitaria representa un impacto grave a los flujos de caja y la rentabilidad de la Corporación Universitaria Lasallista. Este proyecto transforma la postura reactiva frente a los desertores en una postura predictiva y activa, aplicando los lineamientos del Arquetipo de Marca Mago-Ciudadano.")
            with doc2:
                st.markdown("### 2. Tratamiento Ético y Unicidad Cronológica (ETL)")
                st.write("El código integrado aplica un motor lógico ETL que agrupa la base por la llave primaria, ordena los tiempos cronológicamente y aplica el método de aislamiento de estado terminal. De manera innegociable, el sistema descarta los registros pasados y prioriza el estado actual absoluto del individuo.")
            with doc3:
                st.markdown("### 3. Anatomía del Simulador de Escenarios")
                st.write("El simulador financiero opera bajo una función matemática de **Desgaste Proporcional**. La Base Inicial (N) menos la cuota de recuperación proyectada (λ) demuestra crudamente cómo las campañas a largo plazo sobre una base estática sufren de rendimientos marginales decrecientes.")
            with doc4:
                st.markdown("### 4. Sustentación de Algoritmos de Machine Learning")
                st.write("Se empleó **Scikit-Learn (sklearn)** para dotar a la herramienta de capacidades autónomas, incluyendo Regresión Lineal de Mínimos Cuadrados Ordinarios (OLS) para proyectar la curva del mercado y Árbol de Clasificación Gini para perfilar la taxonomía del riesgo sociodemográfico.")
            with doc5:
                st.markdown("### 5. Glosario Oficial y Metadatos del Sistema")
                st.write("""
                * **Target Nivel 5+:** Todo estudiante que superó la mitad técnica de su carrera.
                * **¿ESNUEVO?:** Columna categórica que separa a la población entre "NUEVO" y "ANTIGUO" (retención).
                """)

        # =============================================================================
        # FOOTER INAMOVIBLE (CREDITOS / MARCA / COPYRIGHT)
        # =============================================================================
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #606060; font-size: 15px; padding: 25px 0; background-color: #f1f3f4; border-radius: 10px;">
                <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br><br>
                Desarrollo e Insumo de Investigación aportado por la <strong>Facultad de Ingeniería</strong> bajo la dirección general de <strong>Feibert Alirio Guzmán Pérez</strong>.<br><br>
                <i>Este tablero interactivo predictivo y aplicativo soporta tecnológica y científicamente al proyecto adscrito al grupo de investigación <strong>G-3IN</strong>.</i>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada o contacta al administrador del sistema.")
    st.exception(e)
