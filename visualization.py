import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("📊 Data Visualization")

    # Check if cleaned dataset is available
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No cleaned dataset found! Please complete the Data Cleaning process first.")
        return

    # Load cleaned dataset
    df = st.session_state['dataframe']

    # Sidebar for selecting visualization type
    st.sidebar.header("📌 Select Visualization Type")
    vis_type = st.sidebar.radio(
        "Choose a type:", 
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Advanced Visualizations"]
    )

    # Univariate Analysis (Single Column)
    if vis_type == "Univariate Analysis":
        st.subheader("📈 Univariate Analysis")

        numeric_cols = df.select_dtypes(include=['number']).columns
        selected_col = st.selectbox("📌 Select a Numeric Column", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    # Bivariate Analysis (Two Columns)
    elif vis_type == "Bivariate Analysis":
        st.subheader("🔀 Bivariate Analysis")

        numeric_cols = df.select_dtypes(include=['number']).columns
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("📌 Select X-axis Column", numeric_cols)
        with col2:
            y_col = st.selectbox("📌 Select Y-axis Column", numeric_cols)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)

    # Multivariate Analysis (Multiple Columns)
    elif vis_type == "Multivariate Analysis":
        st.subheader("📊 Multivariate Analysis")

        numeric_cols = df.select_dtypes(include=['number']).columns
        selected_cols = st.multiselect("📌 Select Multiple Columns", numeric_cols)

        if len(selected_cols) >= 2:
            fig, ax = plt.subplots()
            sns.pairplot(df[selected_cols])
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please select at least two columns.")

    # Advanced Visualizations (Pairplot, Heatmap)
    elif vis_type == "Advanced Visualizations":
        st.subheader("📌 Advanced Visualizations")

        plot_type = st.selectbox("Choose Plot Type", ["Pairplot", "Correlation Heatmap"])

        if plot_type == "Pairplot":
            fig = sns.pairplot(df)
            st.pyplot(fig)
        elif plot_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    st.success("✅ Visualization Complete! Modify selections to explore more insights.")
