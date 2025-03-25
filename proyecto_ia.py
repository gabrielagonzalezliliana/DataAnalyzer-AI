import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from groq import Groq

# Configurar API Key de Groq
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
client = Groq(api_key=GROQ_API_KEY)

# App title
st.title('DataAnalyzer AI')

# File uploader
uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Cargar el dataset
    try:
        data = pd.DataFrame(pd.read_csv(uploaded_file))
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    st.write("### Vista Previa de los Datos")
    st.dataframe(data.head())

  

    st.write("### Descripción del DataFrame")
    st.write(data.describe())

   

  

    columns = data.columns
    report_insights = []  # Lista para almacenar insights del reporte final

    # Crear visualizaciones basadas en las columnas disponibles
    for col in columns:
        st.write(f"### Distribución de {col}")
        try:
            if pd.api.types.is_numeric_dtype(data[col]):
                fig = px.histogram(data, x=col, marginal="rug")
                st.plotly_chart(fig)
                explanation = f"Este histograma muestra la distribución de la columna '{col}'."
            else:
                unique_count = data[col].nunique()
                if unique_count < 5:
                    fig = px.pie(data, names=col, title=f"Distribución de {col}")
                    st.plotly_chart(fig)
                    value_counts = data[col].value_counts()
                    if not value_counts.empty:
                        most_frequent_category = value_counts.index[0]
                        most_frequent_count = value_counts[most_frequent_category]
                        explanation = (
                            f"Este gráfico circular muestra la proporción de cada categoría en la columna '{col}'. "
                            f"La categoría más frecuente es '{most_frequent_category}' con {most_frequent_count} ocurrencias."
                        )
                    else:
                        explanation = f"No hay datos para mostrar en la columna '{col}'."
                else:
                    if unique_count > 10:
                        num_categories = st.selectbox(
                            f"Número de categorías a mostrar para {col}",
                            options=[5, 10, 15, 20, "Todas"],
                            key=f"num_categories_{col}"
                        )

                        if num_categories == "Todas":
                            fig = px.bar(data, x=col, title=f"Distribución de {col}")
                            st.plotly_chart(fig)
                        else:
                            top_n = num_categories
                            top_counts = data[col].value_counts().nlargest(top_n).index
                            data_top = data[data[col].isin(top_counts)]
                            fig = px.bar(data_top, x=col, title=f"Top {top_n} categorías de {col}")
                            st.plotly_chart(fig)

                        value_counts = data[col].value_counts()
                        if not value_counts.empty:
                            most_frequent_category = value_counts.index[0]
                            most_frequent_count = value_counts[most_frequent_category]
                            explanation = (
                                f"Este gráfico de barras muestra la distribución de la columna '{col}'. "
                                f"La categoría más frecuente es '{most_frequent_category}' con {most_frequent_count} ocurrencias."
                            )
                        else:
                            explanation = f"No hay datos para mostrar en la columna '{col}'."
                    else:
                        fig = px.bar(data, x=col, title=f"Distribución de {col}")
                        st.plotly_chart(fig)
                        value_counts = data[col].value_counts()
                        if not value_counts.empty:
                            most_frequent_category = value_counts.index[0]
                            most_frequent_count = value_counts[most_frequent_category]
                            explanation = (
                                f"Este gráfico de barras muestra la distribución de la columna '{col}'. "
                                f"La categoría más frecuente es '{most_frequent_category}' con {most_frequent_count} ocurrencias."
                            )
                        else:
                            explanation = f"No hay datos para mostrar en la columna '{col}'."
        except Exception as e:
            st.write(f"No se pudo crear el gráfico para la columna '{col}': {e}")

        # Explicación con IA
        try:
            value_counts = data[col].value_counts(normalize=True) * 100
            categories_info = "\n".join([f"{cat}: {val:.1f}%" for cat, val in value_counts.items()])

            prompt = f"""Analiza los datos representados en el siguiente gráfico.
            Proporciona insights clave basándote únicamente en las siguientes categorías y porcentajes:

            {categories_info}

            No inventes valores ni categorías. Destaca patrones y tendencias en base a los datos proporcionados.
            Responde en español."""

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=300
            )
            detailed_explanation = response.choices[0].message.content
            st.write(f"Explicación con IA: {detailed_explanation}")
            report_insights.append(detailed_explanation)

        except Exception as e:
            st.write(f"No se pudo generar la explicación con IA: {e}")
       

    # Mapa de calor de correlación (si hay más de una columna numérica)
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        st.write("### Mapa de calor de correlación")
        try:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                            labels=dict(x="Columnas", y="Columnas", color="Correlación"),
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            color_continuous_scale="RdBu",
                            text_auto=".2f")
            st.plotly_chart(fig)

            # Encontrar las correlaciones más fuertes
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            strong_correlations = upper_triangle.abs().unstack().sort_values(ascending=False)
            strong_correlations = strong_correlations[strong_correlations > 0.5]

            if not strong_correlations.empty:
                insight = "Este mapa de calor muestra las correlaciones más fuertes entre las variables numéricas:\n"
                for pair, correlation in strong_correlations.head(5).items():
                    insight += f"- {pair[0]} y {pair[1]}: {correlation:.2f}\n"
                report_insights.append(insight)
                st.write(insight)
            else:
                st.write("No hay correlaciones fuertes entre las variables.")

        except Exception as e:
            st.write(f"No se pudo crear el mapa de calor de correlación: {e}")

    # Generar Reporte Final con IA

    # Generar Reporte Final con IA
    st.write("### Reporte Final con IA")
    try:
        prompt = f"""Genera un reporte final basado en los siguientes insights obtenidos del análisis de datos.
        El reporte debe enfocarse en tendencias clave, patrones relevantes y hallazgos importantes sin describir los gráficos.
        Incluye conclusiones accionables y recomendaciones estratégicas si es posible.
        Por favor, responde en español.

        Insights identificados: {'. '.join(report_insights)}"""
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1024
        )
        final_report = response.choices[0].message.content
        st.write(final_report)
    except Exception as e:
        st.write(f"No se pudo generar el reporte final con IA: {e}")

    try:
        # Descargar el reporte final en PDF
        import weasyprint
        from weasyprint import HTML

        html_string = f"""
        <h1>Reporte Final</h1>
        <p>{final_report}</p>
        """

        pdf_file = HTML(string=html_string).write_pdf()

        st.download_button(
            label="Descargar Reporte Final en PDF",
            data=pdf_file,
            file_name="reporte_final.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.write(f"No se pudo generar el reporte en PDF: {e}")
         