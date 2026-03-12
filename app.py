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
# 1. CONFIGURACIÓN CORPORATIVA Y ESTADO DE SESIÓN (PANTALLA EMERGENTE)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inteligencia de Reingresos - Unilasallista", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# Control de sesión para simular ventana de presentación
if 'app_iniciada' not in st.session_state:
    st.session_state['app_iniciada'] = False

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS Y LIMPIEZA BLINDADA (ETL PRINCIPAL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    
    # Limpieza blindada de caracteres corruptos en columnas (Windows/Mac/SPSS encoding fix)
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
    df_crudo = load_data()

    # =============================================================================
    # PANTALLA EMERGENTE DE BIENVENIDA (PRESENTACIÓN INSTITUCIONAL)
    # =============================================================================
    if not st.session_state['app_iniciada']:
        col_img, col_tit = st.columns([1, 4])
        with col_img:
            try: st.image("logoUnilasalle.png", width=180)
            except: pass
        with col_tit:
            st.title("Sistema Integral de Proyección y Recuperación Estudiantil")
            st.markdown("#### Vicerrectoría Financiera | Corporación Universitaria Lasallista")
        
        st.markdown("---")
        st.markdown("""
        ### 📌 Contexto del Proyecto Analítico
        Bienvenido a la plataforma oficial de inteligencia de negocios enfocada en la viabilidad financiera y la retención académica.
        Este sistema procesa de manera automática miles de transacciones académicas (matrículas, aplazamientos, retiros y reingresos) para transformarlas en un **embudo de recuperación comercial**.
        
        #### ¿Qué encontrarás en este software?
        1. **Módulo Operativo (Leads):** Extracción de listas de llamadas depuradas de duplicados para el Call Center.
        2. **Descriptivos y Mapas:** Diagnóstico territorial y de comportamiento académico.
        3. **Modelos Predictivos (IA):** Proyección futura de reingresos orgánicos y volumen de alumnos NUEVOS empleando Scikit-Learn.
        4. **Simulador de Escenarios:** Un motor de proyección donde mides el impacto de tus campañas.
        """)
        
        st.info(f"💾 **Estado de la Base de Datos Creada:** Se han detectado **{len(df_crudo):,} registros transaccionales** listos para ser procesados matemáticamente.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        if col_btn2.button("🚀 Ingresar a la Plataforma Analítica", use_container_width=True):
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
        # PANEL IZQUIERDO: FILTROS INTEGRADOS
        # -----------------------------------------------------------------------------
        try: st.sidebar.image("est.png", use_column_width=True)
        except: pass
        
        st.sidebar.markdown("### 🔍 Buscador de Prospectos")
        busqueda_txt = st.sidebar.text_input("Ingresar Documento o Nombre", "")
        
        st.sidebar.markdown("### ⚙️ Segmentación de Población")
        st.sidebar.info("La segmentación re-entrena todos los algoritmos y gráficos en tiempo real.")
        
        fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df_crudo['FACULTAD'].dropna().unique())))
        progs_disp = sorted(df_crudo[df_crudo['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df_crudo['PROGRAMA'].dropna().unique())
        prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
        est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df_crudo['ESTRATO'].dropna().unique())))
        cohorte_sel = st.sidebar.selectbox("Cohorte (Año de Ingreso)", ["Todos"] + list(sorted(df_crudo['AÑOCOHORTE'].dropna().unique(), reverse=True)))

        df_base = df_crudo.copy()
        if busqueda_txt:
            mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
            df_base = df_base[mask]
        if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
        if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
        if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
        if cohorte_sel != "Todos": df_base = df_base[df_base['AÑOCOHORTE'] == cohorte_sel]

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📞 Gestión Operativa", 
            "📊 Análisis Descriptivo Completo", 
            "⚙️ Simulador de Retorno", 
            "🧠 Predicciones IA (Reingresos y Nuevos)", 
            "📖 Biblioteca Documental (10 Hojas)"
        ])
        
        # =============================================================================
        # 1. GESTIÓN OPERATIVA
        # =============================================================================
        with tab1:
            st.header("Directorio de Prospectos Depurados")
            st.info(f"**Validación de Unicidad:** Partiendo de una base de {len(df_base):,} transacciones bajo el filtro actual, el sistema aisló el historial para no repetir estudiantes. Se encontraron **{len(df_candidatos_finales)} prospectos definitivos** que superaron el Nivel 4 y actualmente están retirados sin haber reingresado posteriormente.")
            
            with st.expander("💡 ¿Cómo operativizar esta tabla?"):
                st.write("""
                * Esta es su "Base de Oro" comercial. Aquí no hay estudiantes matriculados ni graduados.
                * Al presionar "Descargar", obtendrás el archivo Excel/CSV exacto para entregar a los agentes de llamadas.
                * **Ejemplo de uso:** Selecciona en la barra lateral "Estrato 3", y la tabla se actualizará solo con candidatos de estrato 3 a quienes puedes ofrecerles un alivio financiero específico.
                """)
                
            cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
            st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=450)
            st.download_button(
                label="📥 Exportar Base para Call Center (.CSV)",
                data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'),
                file_name="Leads_Depurados_Unilasallista.csv",
                mime="text/csv"
            )

        # =============================================================================
        # 2. ANÁLISIS DESCRIPTIVO COMPLETO
        # =============================================================================
        with tab2:
            st.header("Radiografía de Tendencias y Población")
            
            # Fila 1: KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Volumen Histórico Total", f"{len(df_base):,}")
            k2.metric("Matrículas Nuevas Totales", f"{len(df_base[df_base['¿ESNUEVO'] == 'NUEVO']):,}")
            k3.metric("Prospectos Retirados (Target Nivel 5+)", f"{len(df_candidatos_finales)}")
            k4.metric("Promedio de Nivel", f"{df_base['NIVEL'].mean():.1f}")
            
            st.markdown("---")
            st.markdown("### 1. Evolución Histórica de Matrícula (Nuevos vs Continuidad)")
            
            # Gráfico de líneas longitudinal
            df_trend = df_base.groupby(['PeriodoAcadémico', '¿ESNUEVO']).size().reset_index(name='Cantidad')
            fig_lin = px.line(df_trend, x='PeriodoAcadémico', y='Cantidad', color='¿ESNUEVO', markers=True,
                              title="Tendencia Longitudinal de Ingresos por Semestre", template="plotly_white")
            st.plotly_chart(fig_lin, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 2. Comportamiento Académico y Fricción")
            colA, colB = st.columns(2)
            
            with colA:
                heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
                fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen',
                                              color_continuous_scale='Blues', text_auto=True,
                                              title="Mapa de Calor: Zonas de Fricción Financiera", template="plotly_white")
                fig_heat.update_layout(xaxis_title="Nivel Académico Cursado", yaxis_title="Estado Final de la Transacción")
                st.plotly_chart(fig_heat, use_container_width=True)
                with st.expander("💡 Explicación Descriptiva"):
                    st.write("Identifica visualmente en qué semestres exactos se concentra el 'Estudiante Retirado'. Esto permite a la Vicerrectoría saber dónde colocar los incentivos financieros de prevención.")

            with colB:
                fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='ESTADO', ax=ax_box, palette="pastel")
                ax_box.set_title("Comportamiento de la Deserción según el Estrato", fontweight='bold')
                ax_box.legend(loc='upper right', fontsize='small')
                st.pyplot(fig_box)
                with st.expander("💡 Explicación Descriptiva"):
                    st.write("Demuestra si los estudiantes de bajos ingresos (cajas a la izquierda) desertan de sus carreras mucho más temprano (en niveles bajos) que los estudiantes de mayores ingresos.")

            st.markdown("---")
            st.markdown("### 3. Densidad Territorial de Estudiantes")
            
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
                                            size_max=45, zoom=9, mapbox_style="carto-positron")
                st.plotly_chart(fig_map, use_container_width=True)

        # =============================================================================
        # 3. SIMULADOR DE ESCENARIOS
        # =============================================================================
        with tab3:
            st.header("Motor de Simulación y Conversión")
            st.markdown("Este modelo te permite traducir datos abstractos en estrategias operativas. Aquí asumes que realizarás una campaña activa (por ejemplo, correos masivos o llamadas).")
            
            # Tarjetas EXPLICITAS del Simulador
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
            
            # Tarjetas descriptivas de las operaciones
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Insumo:** \n\nEmpezamos con **{base_ini} prospectos** listos para llamar en la base de datos.")
            c2.success(f"**Proyección:** \n\nCon un {int(tasa_recup*100)}% de éxito, asegurarás **{df_proy_sim['Reingresos'].sum()} nuevas matrículas**.")
            c3.error(f"**Desgaste:** \n\nQuedarán **{base_disp} leads imposibles** de recuperar al final de tu ciclo estratégico.")
            
            # Gráfico de doble eje
            fig_sim, ax_b = plt.subplots(figsize=(12, 5))
            ax_l = ax_b.twinx()
            sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#4C72B0', label="Reingresos (Caja Financiera)")
            sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#C44E52', marker='o', lw=3, label="Base de Contactos Pendientes")
            
            ax_b.set_title("Curva de Decaimiento: Extracción de Valor de la Base de Datos", fontweight='bold')
            ax_b.set_ylabel("Nuevos Reingresados (Cantidad)")
            ax_l.set_ylabel("Teléfonos por contactar", color='#C44E52')
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
            st.markdown("Si la universidad no hace ninguna campaña y se deja llevar por la **tendencia orgánica y la inercia del mercado**, esto es lo que la Inteligencia Artificial prevé que sucederá:")
            
            ia1, ia2, ia3 = st.tabs(["📉 1. Modelo de Reingresos Futuros", "🚀 2. Modelo de NUEVOS ESTUDIANTES", "🌳 3. Árbol de Riesgo Académico"])
            
            # IA 1: REINGRESOS
            with ia1:
                st.markdown("#### Proyección de Retornos Orgánicos")
                tendencia_reing = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Cantidad')
                
                if len(tendencia_reing) > 2:
                    tendencia_reing['Time'] = range(1, len(tendencia_reing) + 1)
                    modelo_r = LinearRegression().fit(tendencia_reing[['Time']], tendencia_reing['Cantidad'])
                    T_fut = pd.DataFrame({'Time': range(tendencia_reing['Time'].max() + 1, tendencia_reing['Time'].max() + 1 + len(per_futuros))})
                    preds_r = [max(0, p) for p in modelo_r.predict(T_fut)]
                    
                    k1, k2 = st.columns(2)
                    k1.metric("Estudiantes Recuperados Orgánicos Proyectados (Próx. 5 años)", f"{int(sum(preds_r))}")
                    k2.info("A partir de los datos históricos aislados, la recta matemática halla el comportamiento natural de retorno sin injerencia publicitaria externa.")
                    
                    fig_r, ax_r = plt.subplots(figsize=(10, 4))
                    sns.regplot(data=tendencia_reing, x='Time', y='Cantidad', ax=ax_r, color="#2CA02C", label="Historia (Con margen estadístico)")
                    ax_r.plot(T_fut['Time'], preds_r, color="#FF7F0E", marker="X", linestyle="--", lw=2, label="Proyección Futura (IA)")
                    ax_r.set_xticks(range(1, len(tendencia_reing) + len(per_futuros) + 1))
                    ax_r.set_xticklabels(list(tendencia_reing['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_r.legend()
                    st.pyplot(fig_r)
                else:
                    st.warning("El historial filtrado no cuenta con el volumen mínimo necesario para entrenar la regresión de Reingresos.")

            # IA 2: NUEVOS ESTUDIANTES (EL MÁS IMPORTANTE)
            with ia2:
                st.markdown("#### Proyección de Captación (Matrícula Nueva)")
                st.markdown("Analizando la columna categórica `¿ESNUEVO == 'NUEVO'`, el algoritmo proyecta la captación general del mercado:")
                
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
                    sns.regplot(data=tendencia_nuevos, x='Time', y='Cantidad', ax=ax_n, color="#1F77B4", label="Historia Real")
                    ax_n.plot(T_fut_n['Time'], preds_n, color="#D62728", marker="o", linestyle="--", lw=2, label="Previsión Futura (IA)")
                    ax_n.set_xticks(range(1, len(tendencia_nuevos) + len(per_futuros) + 1))
                    ax_n.set_xticklabels(list(tendencia_nuevos['PeriodoAcadémico']) + per_futuros, rotation=45)
                    ax_n.set_title("Proyección Regresiva: Comportamiento de Alumnos Nuevos", fontweight='bold')
                    ax_n.legend()
                    st.pyplot(fig_n)
                else:
                    st.warning("Faltan datos de estudiantes nuevos para este filtro particular.")

            # IA 3: ÁRBOL DE DECISIÓN
            with ia3:
                st.markdown("#### Anatomía del Desertor (Riesgo Estadístico)")
                st.write("El árbol extrae reglas de separación en la base de estudiantes inactivos.")
                df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
                if len(df_tree) > 10:
                    df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
                    X = df_tree[['ESTRATO_NUM']].fillna(0)
                    clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
                    clf.fit(X, df_tree['Retiro_Tardío'])
                    
                    fig_tree, ax_t = plt.subplots(figsize=(10, 4), dpi=150)
                    plot_tree(clf, feature_names=['Estrato Socioeconómico'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
                    st.pyplot(fig_tree)
                    st.info("💡 **Cómo leer el árbol:** Los cuadros muestran cómo el algoritmo divide tu población. Si descubres que los cuadros de 'Deserción Tardía' se asocian a la rama de Estratos altos, sabrás que la pérdida económica grave no está en la base de la pirámide.")
                else:
                    st.warning("Poca base de datos para generar el árbol de decisión bajo este filtro.")

        # =============================================================================
        # 5. BIBLIOTECA DOCUMENTAL (NIVEL TESIS / 10 HOJAS VIRTUALES)
        # =============================================================================
        with tab5:
            st.header("Biblioteca de Metodología Analítica")
            st.markdown("Este apartado provee el sustento investigativo, matemático y estructural que hace de esta plataforma una herramienta científicamente válida y no una simple calculadora de Excel.")
            
            doc1, doc2, doc3, doc4, doc5 = st.tabs(["1. Justificación Estratégica", "2. Arquitectura de Datos (ETL)", "3. Ecuación Financiera", "4. Modelado Predictivo (IA)", "5. Glosario de Variables"])
            
            with doc1:
                st.markdown("### 1. Justificación y Fundamentación del Proyecto")
                st.write("""
                La deserción universitaria representa no solo un fracaso en la misión educativa, sino un impacto grave a los flujos de caja y la rentabilidad de la Corporación Universitaria Lasallista. 
                
                Este proyecto de investigación, respaldado por la Facultad de Ingeniería y el grupo G-3IN, transforma la postura reactiva de la universidad frente a los desertores en una postura *predictiva y activa*. Identificar estudiantes de niveles 5 en adelante que no han reingresado permite aplicar campañas de "recuperación de cartera educativa" con altos márgenes de viabilidad, dado que el estudiante avanzado tiene un "costo hundido" (tiempo y dinero invertido) que lo motiva psicológicamente a querer titularse.
                """)
            
            with doc2:
                st.markdown("### 2. Tratamiento Ético y Unicidad Cronológica (ETL)")
                st.write("""
                Un software analítico sin limpieza de datos miente en sus recuentos. Las bases de datos académicas (como SPSS o bases SQL) son **Sistemas Transaccionales (OLTP)**, lo que significa que el mismo alumno genera docenas de filas de registro histórico.
                
                **Resolución de Conflictos:**
                El código integrado en este Dashboard aplica un motor lógico ETL que agrupa la base por la llave primaria (`DOCUMENTOIDENTIDAD`), ordena los tiempos cronológicamente y aplica el método de aislamiento de estado terminal. De manera innegociable, el sistema descarta los registros pasados y prioriza el estado actual absoluto del individuo. Si el estudiante aparece como retirado en 2021 pero como reingreso en 2024, el algoritmo asume el reingreso como estado definitivo y elimina al estudiante de la "Bolsa Comercial" que va hacia el Call Center.
                """)
                
            with doc3:
                st.markdown("### 3. Anatomía del Simulador de Escenarios y Decaimiento")
                st.write("""
                El simulador financiero no opera bajo magia, sino bajo una función matemática de **Desgaste Proporcional o Decaimiento Radiactivo Estratégico**.
                
                1. **La Base Inicial (N):** Es el inventario absoluto y cerrado de estudiantes viables a contactar.
                2. **El Vector de Tasa (λ):** Es el slider interactivo (% de éxito). Si el equipo contacta a toda la base y convence al 15%, logran la primera inyección de caja.
                3. **El Efecto Inventario Finito:** Para el segundo ciclo proyectado, ya no cuentas con N prospectos, sino con N menos los que ya lograste matricular (se resta la cuota). Esta matemática demuestra crudamente en la gráfica cómo las campañas a largo plazo sobre una base estática sufren de rendimientos marginales decrecientes (la línea roja se va a pique).
                """)

            with doc4:
                st.markdown("### 4. Sustentación de Algoritmos de Machine Learning")
                st.write("""
                Se empleó la biblioteca de clase mundial **Scikit-Learn (sklearn)** para dotar a la herramienta de capacidades autónomas:
                * **Regresión Lineal de Mínimos Cuadrados Ordinarios (OLS):** Proyecta la curva del mercado sin intervención universitaria. La franja sombreada (`sns.regplot`) evidencia los Intervalos de Confianza (Error Estándar), lo cual le da validez investigativa e indica que la máquina es consciente de la volatilidad ambiental (pandemias, crisis económica, etc.).
                * **Árbol de Clasificación Gini (`DecisionTreeClassifier`):** Para evitar entrenar modelos en bases desbalanceadas (como tener solo 2 casos de éxito vs 100 fracasos), el algoritmo Gini perfila la taxonomía del riesgo, escindiendo el conjunto de datos donde la pureza de la regla aumenta, descubriendo así los rasgos intrínsecos de los desertores.
                """)
                
            with doc5:
                st.markdown("### 5. Glosario Oficial y Metadatos del Sistema")
                st.write("""
                * **Target Nivel 5+:** Todo estudiante que superó la mitad técnica de su carrera y posee alto índice de recuperación monetaria.
                * **Target de Gestión:** Columna artificial generada por el algoritmo que sobrescribe el `ESTADO` crudo de SPSS con la verdad cronológica innegociable del estudiante.
                * **¿ESNUEVO?:** Columna categórica fundamental que separa a la población entre "NUEVO" (primera matrícula de vida en la institución) y "ANTIGUO" (retención, continuidad de estudios o reingresos históricos). Base vital para el módulo de proyección de captación fresca en el área de IA.
                """)

        # =============================================================================
        # FOOTER INAMOVIBLE (CREDITOS / MARCA / COPYRIGHT)
        # =============================================================================
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center; color: #444; font-size: 15px; padding: 25px 0; background-color: #f1f3f4; border-radius: 10px;">
                <strong>© 2026-1 | Corporación Universitaria Lasallista</strong><br><br>
                Desarrollo e Insumo de Investigación aportado por la <strong>Facultad de Ingeniería</strong> bajo la dirección general de <strong>Feibert Alirio Guzmán Pérez</strong>.<br>
                <i>Este tablero interactivo predictivo y aplicativo soporta tecnológica y científicamente al proyecto adscrito al grupo de investigación <strong>G-3IN</strong>.</i>
            </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada o contacta al administrador del sistema.")
    st.exception(e)
