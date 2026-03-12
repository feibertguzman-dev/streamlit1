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
# CONFIGURACIÓN CORPORATIVA Y ESTÉTICA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sistema de Inteligencia de Reingresos", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

col_logo, col_title = st.columns([1, 4])
with col_logo:
    try: st.image("logoUnilasalle.png", width=180)
    except: st.write("*(Logo Unilasallista)*")
with col_title:
    st.title("Plataforma Analítica y Predictiva de Reingresos")
    st.markdown("#### Vicerrectoría Financiera | Corporación Universitaria Lasallista")
st.markdown("---")

# -----------------------------------------------------------------------------
# CARGA DE DATOS Y LIMPIEZA BLINDADA
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
    # PESTAÑAS DEL SISTEMA
    # -----------------------------------------------------------------------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Inicio",
        "📞 Gestión Operativa", 
        "📊 Análisis Descriptivo", 
        "⚙️ Simulador Financiero", 
        "🧠 Machine Learning", 
        "📖 Documentación Técnica"
    ])
    
    # =============================================================================
    # PROCESO ETL: Unicidad Cronológica y Resolución de Conflictos Transaccionales
    # =============================================================================
    df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
    
    def clasificar_target(estados):
        lista = estados.tolist()
        if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
        elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
        return 'No Aplica'
        
    estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
    df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
    
    # LA BASE DE ORO: Prospectos decantados
    df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]

    # =============================================================================
    # PESTAÑA 0: INICIO / BIENVENIDA
    # =============================================================================
    with tab0:
        st.markdown("## Inteligencia Analítica de Reingresos")
        st.write("Bienvenido(a) a la plataforma oficial de Business Intelligence y Machine Learning de la **Vicerrectoría Financiera**. Esta herramienta transforma el historial plano de datos en estrategias claras para recuperar la matrícula de estudiantes inactivos.")
        
        st.markdown("---")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Registros Históricos Transaccionales", f"{len(df):,}")
        col_k2.metric("Estudiantes Únicos (Unicidad)", f"{df['DOCUMENTOIDENTIDAD'].nunique():,}")
        col_k3.metric("Programas Académicos Mapeados", f"{df['PROGRAMA'].nunique()}")
        col_k4.metric("Facultades Integradas", f"{df['FACULTAD'].nunique()}")
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #f4f6f9; padding: 25px; border-radius: 8px; border-left: 6px solid #1E88E5;">
            <h4 style="color: #1E88E5; margin-top: 0;">Estructura del Sistema (Navegación)</h4>
            <ul style="color: #333; font-size: 15px; line-height: 1.6;">
                <li><strong>📞 Gestión Operativa:</strong> Extrae la lista de contactos filtrada y sin duplicados para iniciar llamadas hoy mismo.</li>
                <li><strong>📊 Análisis Descriptivo:</strong> Radiografía poblacional. Mapas de calor, dispersión de retiros y geolocalización de estudiantes.</li>
                <li><strong>⚙️ Simulador Financiero:</strong> Ajusta la meta de captación comercial y mira cómo se comportará tu presupuesto y tu base de datos a futuro.</li>
                <li><strong>🧠 Machine Learning:</strong> Algoritmos de Scikit-Learn que descubren perfiles de deserción y predicen la inercia orgánica (Regresión Lineal y Árboles).</li>
                <li><strong>📖 Documentación Técnica:</strong> La base teórica del proyecto, reglas del ETL y manual de uso.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # =============================================================================
    # PESTAÑA 1: TABLA OPERATIVA DE CONTACTO
    # =============================================================================
    with tab1:
        st.header("Directorio Operativo de Contacto")
        st.success(f"📌 **Contexto del Filtro:** Analizando un universo de **{df_base['DOCUMENTOIDENTIDAD'].nunique():,}** estudiantes únicos según el menú lateral.")
        st.markdown(f"**Prospectos Objetivo:** El sistema ha filtrado y encontrado **{len(df_candidatos_finales)} prospectos definitivos** (Nivel >= 5, estado Retirado/Aplazado, sin historial de retorno posterior).")
        
        with st.expander("💡 ¿Cómo interpretar y usar este módulo?"):
            st.write("""
            Aquí se materializa el trabajo de la IA. La tabla que ves abajo es tu lista de prospectos limpia. 
            No hay estudiantes activos, ni graduados, ni egresados en esta tabla. 
            Solo presiona **Descargar Listado (.CSV)** y pásalo a tu equipo de admisiones o retención. Si seleccionas la Facultad de Ingeniería en la izquierda, esta tabla solo te dará los números de teléfono de Ingeniería.
            """)
            
        cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
        st.dataframe(df_candidatos_finales[cols_gestion], use_container_width=True, height=400)
        
        st.download_button(
            label="📥 Descargar Listado Depurado (.CSV) para Contacto",
            data=df_candidatos_finales[cols_gestion].to_csv(index=False, sep=";").encode('utf-8-sig'),
            file_name="Leads_Reingresos_Unilasallista.csv",
            mime="text/csv"
        )

    # =============================================================================
    # PESTAÑA 2: ANÁLISIS DESCRIPTIVO (MAPAS Y CALOR)
    # =============================================================================
    with tab2:
        st.header("Radiografía de Permanencia y Mapas")
        
        st.markdown("### 1. Zonas de Fricción Académica (Mapas de Calor)")
        colA, colB = st.columns(2)
        with colA:
            # Mapa de calor usando plotly express
            heat_data = df_base.groupby(['NIVEL', 'ESTADO']).size().reset_index(name='Volumen')
            fig_heat = px.density_heatmap(heat_data, x='NIVEL', y='ESTADO', z='Volumen',
                                          color_continuous_scale='Blues', text_auto=True,
                                          title="Concentración de Estados por Semestre", template="plotly_white")
            fig_heat.update_layout(xaxis_title="Nivel Académico Cursado", yaxis_title="")
            st.plotly_chart(fig_heat, use_container_width=True)
            with st.expander("💡 ¿Cómo leer este Mapa de Calor?"):
                st.write("Los cuadros más oscuros indican dónde hay mayor acumulación de estudiantes. Busca la fila 'Estudiante Retirado' y mira en qué columna (Nivel) está el cuadro más oscuro. Ahí es donde la universidad está perdiendo financieramente a sus estudiantes.")

        with colB:
            # Gráfico de cajas formal Seaborn
            fig_box, ax_box = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=df_base, x='ESTRATO', y='NIVEL', hue='ESTADO', ax=ax_box, palette="Set2")
            ax_box.set_title("Comportamiento de la Deserción según el Estrato", fontweight='bold')
            ax_box.set_ylabel("Nivel de Estudios")
            ax_box.set_xlabel("Estrato Socioeconómico")
            ax_box.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(fig_box)
            with st.expander("💡 ¿Cómo leer este Diagrama de Caja?"):
                st.write("Este gráfico cruza tres variables. Te permite ver si los estudiantes de Estrato 2 se retiran antes (cajas más bajas) que los de Estrato 5 (cajas más altas).")

        st.markdown("---")
        st.markdown("### 2. Mapa Territorial de Mercado")
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
            with st.expander("💡 ¿Cómo leer la Geolocalización?"):
                st.write("Cada burbuja roja representa una ciudad donde viven los estudiantes que filtraste. Entre más grande la burbuja, mayor cantidad de estudiantes. Es vital para dirigir campañas físicas o sectorizar bases de datos.")

    # =============================================================================
    # PESTAÑA 3: SIMULADOR DE ESCENARIOS (PROYECCIÓN GERENCIAL)
    # =============================================================================
    with tab3:
        st.header("Simulador de Retorno Financiero")
        base_ini = len(df_candidatos_finales)
        
        st.markdown(f"**Punto de Partida:** Tienes un inventario actual de **{base_ini} prospectos de alto valor**. Ajusta el control para ver qué pasaría si lograras contactarlos y convencer a un porcentaje de ellos.")
        tasa_recup = st.slider("🎯 Define tu Meta de Conversión (% de reingresos esperados por periodo)", min_value=1.0, max_value=50.0, value=10.0, step=1.0) / 100.0
        
        per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
        base_disp = base_ini
        proy_sim = []
        
        for per in per_futuros:
            reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
            if reing > base_disp: reing = base_disp
            base_disp -= reing
            proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario_Restante': base_disp})
            
        df_proy_sim = pd.DataFrame(proy_sim)
        
        fig_sim, ax_b = plt.subplots(figsize=(12, 5))
        ax_l = ax_b.twinx()
        
        sns.barplot(data=df_proy_sim, x='Periodo', y='Reingresos', ax=ax_b, color='#4C72B0', label="Nuevos Matriculados")
        sns.lineplot(data=df_proy_sim, x='Periodo', y='Inventario_Restante', ax=ax_l, color='#C44E52', marker='o', lw=3, label="Base Pendiente por Contactar")
        
        ax_b.set_title("Curva Financiera: Captación Nueva vs Agotamiento de la Base de Datos", fontweight='bold')
        ax_b.set_ylabel("Cantidad de Retornados")
        ax_l.set_ylabel("Prospectos Restantes", color='#C44E52')
        ax_b.tick_params(axis='x', rotation=45)
        
        lines1, labels1 = ax_b.get_legend_handles_labels()
        lines2, labels2 = ax_l.get_legend_handles_labels()
        ax_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax_l.get_legend().remove()
        st.pyplot(fig_sim)
        
        with st.expander("💡 ¿Cómo interpretar este simulador?"):
            st.write("""
            * **Las Barras Azules** indican el flujo de caja: cuántos estudiantes lograrás matricular en cada semestre futuro si mantienes el equipo trabajando a la tasa elegida.
            * **La Línea Roja** indica tu inventario: a medida que tu equipo llama y recupera personas, la lista de teléfonos se reduce. Cuando la línea roja llegue abajo, habrás agotado por completo tu base histórica.
            """)

    # =============================================================================
    # PESTAÑA 4: MACHINE LEARNING (CON FALLBACK DE DATOS INTELIGENTE)
    # =============================================================================
    with tab4:
        st.header("Modelos Predictivos Orgánicos (Inteligencia Artificial)")
        st.markdown("A diferencia del Simulador (donde *tú* pones la tasa meta), estos modelos usan algoritmos para descubrir qué pasará orgánicamente basándose en las matemáticas del comportamiento histórico.")
        
        st.subheader("1. Predicción Orgánica: Regresión Lineal")
        
        # Extraemos los reingresos de la base transaccional filtrada
        tendencia = df_base[df_base['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos')
        
        # Mecanismo de seguridad: Si el usuario filtra tanto que ya no hay historial, usamos el histórico de toda la universidad
        uso_global = False
        if len(tendencia) <= 2:
            uso_global = True
            tendencia = df[df['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos')
            st.warning("⚠️ **Alerta de Datos Insuficientes:** Los filtros actuales redujeron demasiado el historial y la IA no puede trazar una línea con menos de 3 puntos en el tiempo. *A continuación se muestra la proyección predictiva GLOBAL de toda la universidad.*")

        if len(tendencia) > 2:
            tendencia['Time'] = range(1, len(tendencia) + 1)
            X_train, y_train = tendencia[['Time']], tendencia['Reingresos']
            modelo = LinearRegression().fit(X_train, y_train)
            
            T_fut = pd.DataFrame({'Time': range(tendencia['Time'].max() + 1, tendencia['Time'].max() + 1 + len(per_futuros))})
            preds = [max(0, p) for p in modelo.predict(T_fut)]
            
            fig_reg, ax_reg = plt.subplots(figsize=(12, 5))
            sns.regplot(data=tendencia, x='Time', y='Reingresos', ax=ax_reg, color="#2CA02C", label="Datos Históricos Reales (Con Margen de Error)")
            ax_reg.plot(T_fut['Time'], preds, color="#FF7F0E", marker="X", linestyle="--", linewidth=2.5, markersize=8, label="Predicción Futura ML")
            
            ax_reg.set_xticks(range(1, len(tendencia) + len(per_futuros) + 1))
            ax_reg.set_xticklabels(list(tendencia['PeriodoAcadémico']) + per_futuros, rotation=45)
            ax_reg.set_title("Regresión Lineal: Predicción de Reingresos Orgánicos", fontweight='bold')
            ax_reg.legend()
            st.pyplot(fig_reg)
            
            with st.expander("💡 ¿Cómo leer el Modelo de Regresión?"):
                st.write("""
                * **Línea Verde con Sombra:** Es la Inteligencia Artificial analizando el pasado. La sombra transparente es el **Intervalo de Confianza**, significa que el modelo reconoce que hay cierta volatilidad y varianza natural.
                * **Línea Punteada Naranja:** Es el pronóstico futuro. Te dice cuántos estudiantes van a volver por su propia cuenta si la universidad no invierte en campañas adicionales ni hace ningún esfuerzo de retención nuevo.
                """)
        else:
            st.error("Incluso sin filtros, la base de datos subida no tiene historial suficiente para una regresión.")

        st.markdown("---")
        st.subheader("2. Árbol de Decisión: Perfilado de Riesgo de Deserción")
        st.write("Dado que predecir el reingreso con pocos casos de éxito no es ético a nivel de datos, este algoritmo perfila cómo se dividen sociodemográficamente los estudiantes que se retiran, clasificándolos en Deserción Temprana o Tardía.")
        
        df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
        if len(df_tree) > 10:
            df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
            X = df_tree[['ESTRATO_NUM']].fillna(0)
            
            clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
            clf.fit(X, df_tree['Retiro_Tardío'])
            
            fig_tree, ax_t = plt.subplots(figsize=(10, 5), dpi=150)
            plot_tree(clf, feature_names=['Estrato'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
            st.pyplot(fig_tree)
            
            with st.expander("💡 ¿Cómo leer el Árbol de Decisiones Algorítmico?"):
                st.write("""
                El algoritmo toma la base de desertores y los divide matemáticamente tratando de encontrar un patrón:
                * Mira la caja superior (la raíz). Si dice por ejemplo 'Estrato <= 3.5', significa que la IA descubrió que la principal diferencia radica entre los estratos bajos (1,2,3) y los altos (4,5,6).
                * Sigue el camino ("True" o "False") para ver dónde caen la mayoría de los estudiantes de Deserción Tardía. Así descubres si estás perdiendo a estudiantes avanzados de bajos o altos recursos.
                """)
        else:
            st.warning("No hay suficientes desertores bajo el filtro actual para armar el árbol.")

    # =============================================================================
    # PESTAÑA 5: DOCUMENTACIÓN TÉCNICA Y SUSTENTACIÓN
    # =============================================================================
    with tab5:
        st.header("📖 Manual Técnico y Fundamentación de Reglas")
        
        with st.expander("⚖️ 1. Principio de Unicidad Cronológica (Motor ETL)"):
            st.markdown("""
            **El Reto Transaccional:**
            En las bases de datos de admisiones y registro académico, un mismo individuo genera múltiples transacciones en el tiempo. Si no se tratan, un estudiante aparecerá duplicado en los recuentos totales, falseando el presupuesto en Power BI.
            
            **La Solución Implementada:**
            El sistema cuenta con un procedimiento de limpieza (ETL) automático:
            1. **Aislamiento Terminal:** Se agrupa por número de documento de identidad y se aísla únicamente la transacción correspondiente a la fecha máxima absoluta.
            2. **Anulación Histórica:** El requerimiento innegociable exige que si un estudiante fue "Retirado" pero años después figura como "Reingresó", el modelo debe asumir su éxito y anular su retiro del inventario de prospectos actual.
            """)
            
        with st.expander("🎯 2. Objeto del Proyecto Financiero"):
            st.write("Proveer a la Corporación Universitaria Lasallista de una herramienta gerencial que traduzca bases de datos planas en embudos prospectivos viables, permitiendo planear, ejecutar (mediante la tabla de contactos) y evaluar el retorno de inversión sobre estrategias de reingreso estudiantil de Nivel Superior (Nivel 5 al 10).")

        with st.expander("🧠 3. Coherencia en el Modelo de Machine Learning"):
            st.write("La Inteligencia Artificial no hace magia con datos vacíos. En el Dashboard se implementaron salvaguardas (Safety Fallbacks). Si un usuario filtra la población a un tamaño minúsculo donde la Regresión Lineal matemáticamente colapsa, el algoritmo interrumpe el error, avisa en pantalla, y cambia temporalmente el modelo a la línea de base poblacional global de la universidad para no perder visualización.")

    # =============================================================================
    # CRÉDITOS INVESTIGATIVOS Y PROPIEDAD INTELECTUAL (FOOTER GLOBAL)
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
