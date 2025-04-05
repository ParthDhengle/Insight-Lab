import streamlit as st
import pandas as pd
from PIL import Image

def show():
    st.title("🏠 Welcome to EDA Explorer")
    
    st.write("""
    **EDA Explorer** is an interactive tool that helps you analyze datasets step by step. Follow the structured process:
    
    1️⃣ **Before Cleaning** - Explore the raw dataset.
    
    2️⃣ **Data Cleaning** - Fix missing values, outliers, and data types.
    
    3️⃣ **After Cleaning** - Review the cleaned dataset.
    
    4️⃣ **Visualization** - Generate insightful plots.
    
    5️⃣ **Hypothesis & Report** - Get AI-generated insights & download a PDF report.
    """)
    
    st.subheader("📌 How It Works")

    flowchart = Image.open("assets/eda-FlowChart.png")  # Ensure the correct path
    st.image(flowchart, caption="EDA Process Flow")

    st.write("### 📂 Upload a CSV File to Start")
    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("❌ The uploaded file is empty. Please upload a valid CSV file.")
                return
            
            # Store DataFrame in session state
            st.session_state['uploaded_file'] = uploaded_file  # Store file reference
            st.session_state['original_df'] = df.copy()  # Store a copy of original dataset
            st.session_state['dataframe'] = df.copy() 
            
            st.success("✅ File uploaded successfully! Now navigate to 'Before Cleaning' to explore it.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
