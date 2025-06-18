import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Titanic", layout="wide")

st.title("🚢 Análisis Exploratorio de Datos - Titanic")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df = load_data()

st.subheader("📊 Vista general del dataset")
st.dataframe(df.head())

# Información general
st.subheader("📈 Información básica")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", df.shape[0])
col2.metric("Columnas", df.shape[1])
col3.metric("Nulos totales", df.isnull().sum().sum())

# Valores nulos
st.subheader("❗ Valores nulos por columna")
st.write(df.isnull().sum())

# Estadísticas
st.subheader("📋 Estadísticas descriptivas")
st.write(df.describe())

# Gráficos
st.subheader("📊 Visualizaciones")

with st.expander("Distribución de la edad"):
    fig, ax = plt.subplots()
    sns.histplot(df["Age"].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

with st.expander("Supervivencia por sexo"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Sex", hue="Survived", ax=ax)
    st.pyplot(fig)

with st.expander("Supervivencia por clase"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax)
    st.pyplot(fig)

with st.expander("Mapa de calor de nulos"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Creado con ❤️ usando Streamlit")

