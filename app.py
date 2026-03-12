import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Estudiantes - Unilasallista",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Dashboard de Estudiantes - Universidad La Salle")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("DataSPSSReingreso.csv", sep=";")
    return df

# Cargar datos
try:
    df = load_data()
    
    # Información básica del dataset
    st.header("📋 Información General")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Número de Columnas", len(df.columns))
    with col3:
        st.metric("Programas Únicos", df['PROGRAMA'].nunique())
    
    st.markdown("---")
    
    # Sidebar con filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro por programa
    programas = df['PROGRAMA'].unique()
    programa_seleccionado = st.sidebar.selectbox("Seleccionar Programa", ["Todos"] + list(programas))
    
    # Filtro por año
    años = sorted(df['AÑO'].unique())
    año_seleccionado = st.sidebar.selectbox("Seleccionar Año", ["Todos"] + list(años))
    
    # Filtro por estado
    estados = df['ESTADO'].unique()
    estado_seleccionado = st.sidebar.selectbox("Seleccionar Estado", ["Todos"] + list(estados))
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if programa_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['PROGRAMA'] == programa_seleccionado]
    if año_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['AÑO'] == año_seleccionado]
    if estado_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['ESTADO'] == estado_seleccionado]
    
    st.header(f"📊 Datos Filtrados ({len(df_filtrado)} registros)")
    
    # Mostrar datos en tabla
    st.dataframe(df_filtrado, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Estadísticas y gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Distribución por Género")
        genero_counts = df_filtrado['GENERO'].value_counts()
        fig1, ax1 = plt.subplots()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax1.pie(genero_counts.values, labels=genero_counts.index, autopct='%1.1f%%', colors=colors[:len(genero_counts)])
        ax1.axis('equal')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("🏢 Distribución por Estado")
        estado_counts = df_filtrado['ESTADO'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=estado_counts.values, y=estado_counts.index, ax=ax2, palette="viridis")
        ax2.set_xlabel("Cantidad")
        ax2.set_ylabel("Estado")
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # Más estadísticas
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📚 Programas con más estudiantes")
        programa_counts = df_filtrado['PROGRAMA'].value_counts().head(10)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=programa_counts.values, y=programa_counts.index, ax=ax3, palette="magma")
        ax3.set_xlabel("Cantidad de Estudiantes")
        ax3.set_ylabel("Programa")
        st.pyplot(fig3)
    
    with col4:
        st.subheader("📍 Top 10 Ciudades")
        ciudad_counts = df_filtrado['CIUDADRESIDENCIA'].value_counts().head(10)
        fig4, ax4 = plt.subplots()
        sns.barplot(x=ciudad_counts.values, y=ciudad_counts.index, ax=ax4, palette="coolwarm")
        ax4.set_xlabel("Cantidad de Estudiantes")
        ax4.set_ylabel("Ciudad")
        st.pyplot(fig4)
    
    st.markdown("---")
    
    # Distribución por estrato
    st.subheader("🏠 Distribución por Estrato")
    estrato_counts = df_filtrado['ESTRATO'].value_counts()
    fig5, ax5 = plt.subplots()
    sns.barplot(x=estrato_counts.values, y=estrato_counts.index, ax=ax5, palette="Blues_d")
    ax5.set_xlabel("Cantidad de Estudiantes")
    ax5.set_ylabel("Estrato")
    st.pyplot(fig5)
    
    # Distribución por año
    st.subheader("📅 Estudiantes por Año")
    año_counts = df_filtrado['AÑO'].value_counts().sort_index()
    fig6, ax6 = plt.subplots()
    sns.lineplot(x=año_counts.index, y=año_counts.values, marker='o', ax=ax6, linewidth=2)
    ax6.set_xlabel("Año")
    ax6.set_ylabel("Cantidad de Estudiantes")
    ax6.set_xticks(año_counts.index)
    st.pyplot(fig6)
    
    st.markdown("---")
    st.markdown("**Datos proporcionados por Universidad La Salle**")
    
except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.info("Asegúrate de que el archivo DataSPSSReingreso.csv esté en el mismo directorio que app.py")
