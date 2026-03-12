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
    # Corrección clave: Renombrar la columna corrupta del Año de Cohorte
    df = df.rename(columns=lambda x: x.replace('AÑ‘OCOHORTE', 'AÑOCOHORTE').strip())
    
    df['NIVEL'] = pd.to_numeric(df['NIVEL'], errors='coerce').fillna(0)
    df['ESTRATO_NUM'] = df['ESTRATO'].astype(str).str.extract('(\d+)').astype(float).fillna(0)
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
    
    # Ahora lee la columna corregida 'AÑOCOHORTE'
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

    st.success(f"📌 **Datos Activos:** Mostrando registros correspondientes a **{df_base['DOCUMENTOIDENTIDAD'].nunique():,}** estudiantes únicos bajo los filtros seleccionados.")

    # -----------------------------------------------------------------------------
    # PESTAÑAS DEL SISTEMA
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📞 Gestión y Contacto (Leads)", 
        "📊 Análisis Descriptivo & Mapa", 
        "⚙️ Simulador de Escenarios", 
        "🧠 Modelos Predictivos (IA)", 
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
    # PESTAÑA 1: TABLA OPERATIVA DE CONTACTO
    # =============================================================================
    with tab1:
        st.header("Directorio Operativo de Contacto")
        st.markdown(f"**Estudiantes Objetivo:** {len(df_candidatos_finales)} candidatos encontrados (Retirados/Aplazados desde Nivel 5).")
        
        with st.expander("💡 ¿Cómo usar esta tabla de gestión?"):
            st.write("""
            * **Objetivo:** Extraer la lista de estudiantes viables para reingreso.
            * **Funcionamiento:** Esta tabla responde a los filtros de la izquierda. Si filtras por Estrato 3 o por el programa Zootecnia, esta tabla solo mostrará esos perfiles.
            * **Acción:** Haz clic en el botón de abajo para descargar el archivo CSV, el cual puedes entregar a admisiones o call center para iniciar el contacto.
            """)
            
        cols_gestion = ['DOCUMENTOIDENTIDAD', 'NOMBRE', 'TELEFONO', 'CELULAR', 'EMAIL', 'PROGRAMA', 'NIVEL', 'ESTRATO', 'CIUDADRESIDENCIA']
        df_mostrar = df_candidatos_finales[cols_gestion].copy()
        
        st.dataframe(df_mostrar, use_container_width=True, height=400)
        
        st.download_button(
            label="📥 Descargar Listado para Call Center (.CSV)",
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
            with st.expander("💡 ¿Cómo leer este gráfico?"):
                st.write("La caja muestra dónde se ubica la mayoría de los estudiantes para ese estado. Si la caja 'Retirado' de hombres se ubica más arriba, significa que los hombres desertan en semestres más avanzados que las mujeres.")

        with colB:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df_base, y='ESTRATO', hue='ESTADO', ax=ax_hist, palette="deep")
            ax_hist.set_title("Volumen Sociodemográfico", fontweight='bold')
            ax_hist.set_ylabel("")
            st.pyplot(fig_hist)
            with st.expander("💡 ¿Cómo leer este gráfico?"):
                st.write("Compara el tamaño total de la población activa vs. retirada en cada estrato para visualizar dónde está el mayor riesgo financiero.")

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
            with st.expander("💡 ¿Cómo leer el mapa?"):
                st.write("Muestra la concentración de la población filtrada según la ciudad registrada en el sistema. Los círculos rojos y grandes representan mayor densidad poblacional. Sirve para dirigir campañas de mercadeo físico.")

    # =============================================================================
    # PESTAÑA 3: SIMULADOR DE ESCENARIOS
    # =============================================================================
    with tab3:
        st.header("Simulador de Retorno Financiero (Estrategia Activa)")
        base_ini = len(df_candidatos_finales)
        
        st.markdown(f"El simulador parte de **{base_ini} estudiantes** listos para ser contactados. Mueve el control deslizante para establecer tu meta de conversión:")
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
        
        with st.expander("💡 ¿Cómo interpretar esta Proyección Financiera?"):
            st.write("""
            Este es un modelo determinístico para planear presupuesto.
            * **Barras Azules:** Muestran cuántos estudiantes vas a matricular en cada semestre si cumples tu tasa de éxito objetivo.
            * **Línea Roja:** Es tu 'inventario' de números de teléfono. Se va acabando porque conforme más gente reingresa, menos personas tienes para llamar en el siguiente periodo.
            """)

    # =============================================================================
    # PESTAÑA 4: MACHINE LEARNING (CON OBJETIVIDAD Y RIGOR)
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
            with st.expander("💡 ¿Cómo interpretar la proyección de regresión?"):
                st.write("La línea verde es el pasado, y su 'sombra' transparente es el margen de error del algoritmo calculado sobre variaciones previas. La línea naranja punteada muestra cuántos estudiantes reingresarán por pura inercia en el futuro si la universidad mantiene las condiciones actuales.")
        else:
            st.warning("No hay suficientes semestres con historial de reingresos bajo estos filtros para trazar la línea de regresión.")

        st.markdown("---")
        st.subheader("2. Árbol de Decisión: Perfil de Deserción")
        st.info("Dado que la base de datos presenta muy pocos reingresos históricos para entrenar una IA confiable, este árbol de decisión inteligente analiza los factores sociodemográficos que causan la deserción, dividiéndola en Temprana (Nivel 1 al 4) y Tardía (Nivel 5 al 10).")
        
        df_tree = df_univ[df_univ['ESTADO'].isin(['Estudiante Retirado', 'Canceló Periodo'])].copy()
        if len(df_tree) > 10:
            df_tree['Retiro_Tardío'] = np.where(df_tree['NIVEL'] >= 5, 1, 0)
            X = df_tree[['ESTRATO_NUM']].fillna(0)
            
            clf = DecisionTreeClassifier(max_depth=2, class_weight='balanced', random_state=42)
            clf.fit(X, df_tree['Retiro_Tardío'])
            
            fig_tree, ax_t = plt.subplots(figsize=(8, 4), dpi=150)
            plot_tree(clf, feature_names=['Estrato'], class_names=['Deserción Temprana', 'Deserción Tardía (Target)'], filled=True, rounded=True, ax=ax_t)
            st.pyplot(fig_tree)
            with st.expander("💡 ¿Cómo interpretar este Árbol de Reglas?"):
                st.write("El sistema evalúa las reglas. Por ejemplo, si en la caja superior dice 'Estrato <= 3.5', significa que el algoritmo separó a los estratos 1, 2 y 3 de los estratos 4, 5 y 6. Sigue las ramas hacia abajo para descubrir si el retiro tardío (nuestro Target) se concentra en los estratos bajos o en los altos según el color y proporción de la caja.")

    # =============================================================================
    # PESTAÑA 5: DOCUMENTACIÓN EXHAUSTIVA Y DESPLEGABLES
    # =============================================================================
    with tab5:
        st.header("📖 Manual Interactivo del Dashboard")
        
        with st.expander("🎯 ¿Cuál es el propósito del sistema?"):
            st.write("El Dashboard es la plataforma analítica de la Vicerrectoría Financiera. Está diseñado para predecir y planear la captación de cartera en estudiantes desertores. Transforma una base de datos plana en un embudo de ventas y proyecciones presupuestales.")
            
        with st.expander("⚙️ ¿Cómo aplican los filtros (Panel Izquierdo)?"):
            st.write("Al seleccionar un programa (Ej. Zootecnia), todo el sistema ignora al resto de la universidad. La tabla de gestión se reduce a candidatos de Zootecnia, el simulador recalcula el dinero recuperable solo para Zootecnia, y el modelo de Machine Learning re-evalúa el comportamiento exclusivo de esa carrera.")
            
        with st.expander("📊 ¿Qué es la 'Gestión y Contacto'?"):
            st.write("Es la función operativa del dashboard. Toma las reglas de negocio (Solo alumnos de Nivel 5 o superior que estén en estado Retirado/Aplazado y que nunca hayan reingresado) y extrae sus correos y teléfonos listos para ser entregados al Call Center.")

        with st.expander("🧠 Ética del Machine Learning implementado"):
            st.write("Entrenar un modelo de Inteligencia Artificial para que prediga 'qué alumno va a reingresar' cuando en el histórico solo hay 2 alumnos que lo han hecho frente a cientos que no, es forzar a la máquina a mentir. Por ello, la IA se calibró para encontrar correlaciones descriptivas reales y útiles, como los mapas territoriales y los factores de abandono tardío.")

except Exception as e:
    st.error("Error crítico en la ejecución del Dashboard. Verifica los datos de entrada.")
    st.exception(e)
