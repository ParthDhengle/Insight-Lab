import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def show():
    st.title("📊 Before Data Cleaning")

    # Check if dataset exists in session state
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No file uploaded! Please go to the Home page and upload a CSV file first.")
        return

    df = st.session_state['dataframe']

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Overview")
    st.write(df.describe())

    st.subheader("📊 Dataset Info")
    # Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader("❗ Missing Values")
    missing_values = df.isnull().sum()

    if missing_values.sum() == 0:
        st.success("✅ No missing values detected!")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=missing_values.index, y=missing_values.values, ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Columns")
        plt.ylabel("Missing Values Count")
        st.pyplot(fig)

    st.write("➡️ **Proceed to the 'Data Cleaning' page to clean the dataset!**")
