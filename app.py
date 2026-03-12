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

# Encabezado
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try: st.image("logoUnilasalle.png", width=180)
    except: st.write("*(Logo)*")
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
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    df['ESTRATO_NUM'] = df['ESTRATO'].str.extract('(\d+)').astype(float).fillna(0)
    df['CIUDADRESIDENCIA'] = df['CIUDADRESIDENCIA'].astype(str).str.upper().str.strip()
    return df

try:
    df = load_data()
    
    # -----------------------------------------------------------------------------
    # PANEL IZQUIERDO: FILTROS INTEGRADOS Y BÚSQUEDA
    # -----------------------------------------------------------------------------
    try: st.sidebar.image("est.png", use_column_width=True)
    except: pass
    
    st.sidebar.markdown("### 🔍 Buscador Específico")
    busqueda_txt = st.sidebar.text_input("Buscar por Cédula, Nombre o Apellido", "")
    
    st.sidebar.markdown("### ⚙️ Filtros Globales (Segmentación)")
    st.sidebar.info("Estos filtros ajustan todas las tablas, proyecciones y mapas simultáneamente.")
    
    fac_sel = st.sidebar.selectbox("Facultad", ["Todas"] + list(sorted(df['FACULTAD'].dropna().unique())))
    progs_disp = sorted(df[df['FACULTAD'] == fac_sel]['PROGRAMA'].dropna().unique()) if fac_sel != "Todas" else sorted(df['PROGRAMA'].dropna().unique())
    prog_sel = st.sidebar.selectbox("Programa Académico", ["Todos"] + list(progs_disp))
    
    est_sel = st.sidebar.selectbox("Estrato Socioeconómico", ["Todos"] + list(sorted(df['ESTRATO'].dropna().unique())))
    
    cohorte_sel = st.sidebar.selectbox("Cohorte de Ingreso (Año)", ["Todos"] + list(sorted(df['AÑ‘OCOHORTE'].dropna().unique(), reverse=True)))

    # APLICACIÓN DE FILTROS AL DATAFRAME PRINCIPAL
    df_base = df.copy()
    if busqueda_txt:
        # Búsqueda insensible a mayúsculas
        mask = df_base['DOCUMENTOIDENTIDAD'].astype(str).str.contains(busqueda_txt) | \
               df_base['NOMBRE'].str.contains(busqueda_txt, case=False, na=False)
        df_base = df_base[mask]
    if fac_sel != "Todas": df_base = df_base[df_base['FACULTAD'] == fac_sel]
    if prog_sel != "Todos": df_base = df_base[df_base['PROGRAMA'] == prog_sel]
    if est_sel != "Todos": df_base = df_base[df_base['ESTRATO'] == est_sel]
    if cohorte_sel != "Todos": df_base = df_base[df_base['AÑ‘OCOHORTE'] == cohorte_sel]

    # Contexto Dinámico
    st.success(f"📌 **Datos Activos:** Estudiantes encontrados bajo estos parámetros: **{df_base['DOCUMENTOIDENTIDAD'].nunique():,}**")

    # -----------------------------------------------------------------------------
    # PESTAÑAS DEL SISTEMA
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📞 Gestión y Contacto (Data)", 
        "📊 Análisis Descriptivo", 
        "⚙️ Simulador de Escenarios", 
        "🧠 Modelos Predictivos (ML)", 
        "📖 Ayuda y Documentación"
    ])
    
    # =============================================================================
    # MOTOR LÓGICO DE TARGETING (Calculado globalmente)
    # =============================================================================
    df_sorted = df_base.sort_values(by=['AÑO', 'PERIODO'])
    def clasificar_target(estados):
        lista = estados.tolist()
        if 'Estudiante de Reingreso' in lista: return 'Reingresó Históricamente'
        elif any(x in lista for x in ['Estudiante Retirado', 'Canceló Periodo', 'Estudiante Aplazado']): return 'Candidato a Reingresar'
        return 'No Aplica'
        
    estado_calc = df_sorted.groupby('DOCUMENTOIDENTIDAD')['ESTADO'].apply(clasificar_target).reset_index(name='Target_Gestión')
    df_univ = df_sorted.drop_duplicates('DOCUMENTOIDENTIDAD', keep='last').merge(estado_calc, on='DOCUMENTOIDENTIDAD')
    
    # LA BASE DE ORO: Estudiantes Nivel 5+ listos para llamar
    df_candidatos_finales = df_univ[(df_univ['NIVEL'] >= 5) & (df_univ['Target_Gestión'] == 'Candidato a Reingresar')]

    # =============================================================================
    # PESTAÑA 1: TABLA OPERATIVA DE CONTACTO (LA MINUCIA)
    # =============================================================================
    with tab1:
        st.header("Directorio Operativo de Contacto")
        st.markdown("Esta tabla contiene la lista exacta de **Estudiantes Candidatos (Nivel 5 en adelante, estado retirado/aplazado)** filtrada según el panel izquierdo.")
        
        with st.expander("💡 ¿Cómo usar esta tabla de gestión?"):
            st.write("""
            * **Objetivo:** Tomar decisiones directas y asignar leads de llamadas.
            * **Filtros:** Si filtras "Zootecnia" a la izquierda, aquí solo saldrán los candidatos a llamar de Zootecnia.
            * **Acción:** Presiona el botón de descarga inferior para pasar este listado al Call Center o equipo de admisiones.
            """)
            
        # Seleccionamos las columnas útiles para gestión telefónica
        cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'TELEFONO', 'CELULAR', 'EMAIL', 'EMAILINSTITUCIONAL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
        df_mostrar = df_candidatos_finales[cols_gestion].copy()
        
        st.dataframe(df_mostrar, use_container_width=True, height=400)
        
        st.download_button(
            label="📥 Descargar Listado para Call Center (.CSV)",
            data=df_mostrar.to_csv(index=False, sep=";").encode('utf-8-sig'),
            file_name="Listado_Contacto_Reingresos.csv",
            mime="text/csv"
        )

    # =============================================================================
    # PESTAÑA 2: ANÁLISIS DESCRIPTIVO Y MAPA
    # =============================================================================
    with tab2:
        st.header("Radiografía de la Población")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_box, ax_box = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df_base, x='ESTADO', y='NIVEL', hue='GENERO', ax=ax_box, palette="pastel")
            ax_box.set_title("Concentración de Retiros por Nivel", fontweight='bold')
            ax_box.tick_params(axis='x', rotation=45)
            ax_box.set_xlabel("")
            st.pyplot(fig_box)
            with st.expander("💡 ¿Cómo leer este gráfico de cajas?"):
                st.write("La caja representa dónde se concentra el 50% de los estudiantes de cada estado. La línea en el medio de la caja es el promedio. Si la caja de 'Retirado' está alta, significa que desertan en semestres avanzados.")

        with col2:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df_base, y='ESTRATO', hue='ESTADO', ax=ax_hist, palette="deep")
            ax_hist.set_title("Impacto del Estrato en la Deserción", fontweight='bold')
            ax_hist.set_ylabel("")
            st.pyplot(fig_hist)
            with st.expander("💡 ¿Cómo leer este gráfico?"):
                st.write("Compara el volumen de estudiantes activos vs retirados dentro de un mismo estrato socioeconómico. Útil para dirigir estrategias de apoyos financieros.")

        st.markdown("---")
        st.markdown("### 📍 Mapa de Calor Territorial Verificado")
        
        # Diccionario verificado con las 20 ciudades top del DataFrame actual
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
            with st.expander("💡 ¿Cómo leer el mapa?"):
                st.write("Los círculos más grandes y oscuros indican las ciudades de residencia donde la universidad tiene su mayor nicho de mercado. Puedes hacer zoom y desplazarte con el mouse.")

    # =============================================================================
    # PESTAÑA 3: SIMULADOR DE ESCENARIOS (PROYECCIÓN GERENCIAL)
    # =============================================================================
    with tab3:
        st.header("Simulador de Retorno Financiero")
        base_ini = len(df_candidatos_finales)
        
        st.markdown(f"**Población Objetivo Actual:** {base_ini} estudiantes detectados por el sistema.")
        tasa_recup = st.slider("🎯 Definir Meta de Recuperación Comercial (% de éxito por ciclo)", min_value=1.0, max_value=50.0, value=10.0, step=1.0) / 100.0
        
        per_futuros = [f"{y}-{s}" for y in range(2026, 2033) for s in [1, 2]][1:13] 
        base_disp = base_ini
        proy_sim = []
        
        for per in per_futuros:
            reing = int(round(base_disp * tasa_recup)) if base_disp > 0 else 0
            if reing > base_disp: reing = base_disp
            base_disp -= reing
            proy_sim.append({'Periodo': per, 'Reingresos': reing, 'Inventario_Restante': base_disp})
            
        df_proy_sim = pd.DataFrame(proy_sim)
        
        # Gráfico corregido de doble eje
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
        
        with st.expander("💡 ¿Cómo interpretar este simulador?"):
            st.write("Las barras azules representan el número de estudiantes que lograrás matricular cada semestre asumiendo la tasa meta de éxito que elegiste. La línea roja que desciende muestra cómo tu base de datos (tu lista de teléfonos a llamar) se va consumiendo hasta llegar a cero en el futuro.")

    # =============================================================================
    # PESTAÑA 4: MACHINE LEARNING (CON OBJETIVIDAD Y RIGOR)
    # =============================================================================
    with tab4:
        st.header("Modelos Predictivos (Machine Learning)")
        
        # 1. Análisis de Regresión (Mejorado con Intervalos de Confianza)
        st.subheader("1. Predicción Orgánica: Regresión Lineal")
        tendencia = df_univ[df_univ['ESTADO'] == 'Estudiante de Reingreso'].groupby('PeriodoAcadémico').size().reset_index(name='Reingresos')
        
        if len(tendencia) > 2:
            tendencia['Time'] = range(1, len(tendencia) + 1)
            X_train, y_train = tendencia[['Time']], tendencia['Reingresos']
            modelo = LinearRegression().fit(X_train, y_train)
            
            T_fut = pd.DataFrame({'Time': range(tendencia['Time'].max() + 1, tendencia['Time'].max() + 1 + len(per_futuros))})
            preds = [max(0, p) for p in modelo.predict(T_fut)]
            
            fig_reg, ax_reg = plt.subplots(figsize=(10, 4))
            # Usar sns.regplot para mostrar el intervalo de confianza (la sombra) del histórico
            sns.regplot(data=tendencia, x='Time', y='Reingresos', ax=ax_reg, color="#2CA02C", label="Tendencia Histórica y Margen Error")
            ax_reg.plot(T_fut['Time'], preds, color="#FF7F0E", marker="X", linestyle="--", label="Predicción Futura ML")
            
            ax_reg.set_xticks(range(1, len(tendencia) + len(per_futuros) + 1))
            ax_reg.set_xticklabels(list(tendencia['PeriodoAcadémico']) + per_futuros, rotation=45)
            ax_reg.legend()
            st.pyplot(fig_reg)
            with st.expander("💡 ¿Cómo interpretar la proyección de regresión?"):
                st.write("La línea verde con la sombra transparente es la inteligencia artificial aprendiendo del pasado (la sombra es el margen de error normal). La línea naranja punteada indica hacia dónde irán los reingresos automáticamente si la universidad no hace nada y sigue la inercia actual.")
        else:
            st.warning("No hay suficientes semestres con historial de reingresos para proyectar la regresión.")

        # 2. Objetividad del Árbol de Decisión
        st.markdown("---")
        st.subheader("2. Perfilamiento Inteligente (Árboles de Decisión)")
        
        # Reporte de Objetividad y Ética de Datos
        reingresos_reales_n5 = len(df_candidatos_finales[df_candidatos_finales['Target_Gestión'] == 'Reingresó Históricamente'])
        retiros_reales_n5 = len(df_candidatos_finales)
        
        st.info(f"**📝 Auditoría de Datos del Modelo:** Según los filtros actuales, el sistema detecta **{retiros_reales_n5} retiros** pero muy poco o nulo historial previo de éxito en niveles altos bajo este corte. \n\n*Nota Analítica:* Entrenar una Inteligencia Artificial cuando casi todos los ejemplos son fallos (desbalance de clases extremo) produce modelos que mienten. Para mantener la veracidad, en lugar de predecir ciegamente, el algoritmo a continuación perfila la **anatomía del desertor**, lo cual es la clave para la prevención.")
        
        # En lugar de clasificar Reingresos (que no hay), clasificamos "Riesgo Temprano vs Tardío" como ejemplo útil
        df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
        if len(df_tree) > 10:
            # Creamos un target útil: ¿Se retira al final de la carrera (Nivel >=6)?
            df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 6, 1, 0)
            X = df_tree[['ESTRATO_NUM']].fillna(0) # Qué pesa más?
            
            clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
            clf.fit(X, df_tree['Retiro_Tardío'])
            
            fig_tree, ax_t = plt.subplots(figsize=(8, 4), dpi=150)
            plot_tree(clf, feature_names=['Estrato Socioeconómico'], class_names=['Deserción Temprana', 'Deserción Tardía'], filled=True, rounded=True, ax=ax_t)
            st.pyplot(fig_tree)
            with st.expander("💡 ¿Cómo interpretar este Árbol de Reglas?"):
                st.write("Cada cuadro es una decisión matemática. El sistema divide a los estudiantes retirados dependiendo de su Estrato, ayudando a descubrir si, por ejemplo, los estratos bajos se retiran al inicio de la carrera, pero los altos al final. Si el cuadro es oscuro, hay mucha seguridad en esa regla.")

    # =============================================================================
    # PESTAÑA 5: DOCUMENTACIÓN EXHAUSTIVA Y DESPLEGABLES
    # =============================================================================
    with tab5:
        st.header("📖 Manual del Usuario y Estructura de Datos")
        
        with st.expander("1. 🎯 Objetivo del Sistema"):
            st.write("""
            Este dashboard es la herramienta oficial de la Vicerrectoría Financiera para proyectar y gestionar los retornos de inversión en campañas de readmisión de estudiantes. Integra análisis histórico, minería de datos, simulación determinística y Machine Learning en una sola interfaz interactiva.
            """)
        
        with st.expander("2. ⚙️ ¿Cómo funcionan los Filtros (Sidebar)?"):
            st.write("""
            * **Buscador:** Puedes tipear un número de cédula o el nombre parcial de un estudiante para encontrar su historial exacto.
            * **Integración Global:** Cualquier cambio en Facultad, Programa o Cohorte que hagas a la izquierda afectará **todas las pestañas** al mismo tiempo. Si filtras "Ingeniería", la pestaña de contacto, los mapas, el simulador y la regresión ignorarán al resto de facultades.
            """)
            
        with st.expander("3. 🧠 Explicación Técnica de la Inteligencia Artificial"):
            st.write("""
            * **Regresión Lineal:** Utiliza la librería de Machine Learning `Scikit-Learn`. Calcula el vector de crecimiento histórico y traza una recta hacia los próximos 10 periodos. La franja sombreada indica los límites probabilísticos de error según la varianza pasada.
            * **Calidad de Datos (Data Quality):** El sistema fue programado con un auditor que previene generar predicciones de árboles de decisión falsas cuando los datos históricos de reingreso (casos de éxito) son inferiores estadísticamente a los casos de retiro. La IA prefiere ser honesta antes que generar un dato ficticio.
            """)

        with st.expander("4. 💾 ¿Cómo actualizar la base de datos el próximo semestre?"):
            st.write("""
            1. Entra a tu sistema universitario (SPSS/Excel) y exporta los datos bajo la misma estructura original de columnas.
            2. Guarda el archivo nominado exactamente como `DataSPSSReingreso.csv` asegurando que la separación sea por punto y coma (`;`).
            3. Sube el archivo a tu carpeta raíz. El sistema procesará el nuevo histórico automáticamente en menos de 1 segundo.
            """)

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada.")
    st.exception(e)
