import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from datetime import datetime
import google.generativeai as genai
import numpy as np

# Initialize Gemini API
GENAI_API_KEY = "AIzaSyDWqKiiG3etvFCVIk4_GTuiVTvqK45VUrc"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

def call_gemini_api(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error calling Gemini API]: {e}"

# Helper Functions

def generate_dataset_overview(df):
    rows, cols = df.shape
    prompt = f"Analyze the dataset with shape ({rows}, {cols}) and describe its general characteristics."
    insight = call_gemini_api(prompt)
    return {"title": "Dataset Overview", "text": insight}

def generate_missing_value_section(df):
    missing_df = df.isnull().sum().to_frame('Missing Count')
    missing_df['% Missing'] = (missing_df['Missing Count'] / len(df)) * 100
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('% Missing', ascending=False)
    
    if not missing_df.empty:
        fig, ax = plt.subplots()
        sns.barplot(x=missing_df.index, y=missing_df['% Missing'], ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plot_path = "plots/missing_values.png"
        plt.savefig(plot_path)
        plt.close()
    else:
        plot_path = None
    
    prompt = f"Analyze missing value summary: {missing_df.to_string()}"
    insight = call_gemini_api([prompt])[0]
    return {"title": "Missing Value Analysis", "text": insight, "image": plot_path}


def generate_outlier_section(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(6, len(numeric_cols)*2))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Outliers in {col}")
    plt.tight_layout()
    plot_path = "plots/outliers.png"
    plt.savefig(plot_path)

    prompt = f"Analyze these columns for potential outliers: {', '.join(numeric_cols)}"
    insight = call_gemini_api(prompt)

    return {
        "title": "Outlier Analysis",
        "text": insight,
        "image": plot_path
    }

def generate_univariate_visuals(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    prompts = [f"Provide insights on the distribution of {col}." for col in numeric_cols]
    insights = call_gemini_api(prompts)
    
    sections = []
    for col, insight in zip(numeric_cols, insights):
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        plot_path = f"plots/univariate_{col}.png"
        plt.savefig(plot_path)
        plt.close()
        sections.append({"title": f"Univariate Analysis - {col}", "text": insight, "image": plot_path})
    
    return sections

def generate_bivariate_visuals(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pairs = [(col1, col2) for i, col1 in enumerate(numeric_cols) for col2 in numeric_cols[i+1:]]
    prompts = [f"Analyze the relationship between {col1} and {col2}." for col1, col2 in pairs]
    insights = call_gemini_api(prompts)
    
    sections = []
    for (col1, col2), insight in zip(pairs, insights):
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
        ax.set_title(f"Bivariate Analysis: {col1} vs {col2}")
        plt.tight_layout()
        plot_path = f"plots/bivariate_{col1}_{col2}.png"
        plt.savefig(plot_path)
        plt.close()
        sections.append({"title": f"Bivariate Analysis - {col1} vs {col2}", "text": insight, "image": plot_path})
    
    return sections

def generate_multivariate_visuals(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plot_path = "plots/multivariate.png"
    plt.savefig(plot_path)
    plt.close()
    
    prompt = f"Analyze correlation matrix:\n{corr.to_string()}"
    insight = call_gemini_api([prompt])[0]
    return {"title": "Multivariate Analysis", "text": insight, "image": plot_path}

def generate_key_insights(df):
    prompt = f"Based on the EDA of this dataset with {df.shape[0]} rows and {df.shape[1]} columns, generate key hypotheses and next step recommendations."
    return call_gemini_api(prompt)

def save_pdf(sections, filename="eda_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 10, "Exploratory Data Analysis Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    for section in sections:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, section['title'], ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, section['text'])
        if 'image' in section:
            try:
                pdf.image(section['image'], w=160)
                pdf.ln(5)
            except:
                pass
        pdf.ln(10)

    pdf.output(filename)
    return filename

def show():
    st.title("📑 Hypothesis & AI Report")

    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No cleaned dataset found! Please complete Data Cleaning first.")
        return

    df = st.session_state['dataframe']

    st.subheader("📊 Generating Report Sections with AI")
    with st.spinner("Generating sections..."):
        sections = [
            generate_dataset_overview(df),
            generate_missing_value_section(df),
            generate_outlier_section(df)
        ]
        sections.extend(generate_univariate_visuals(df))
        sections.extend(generate_bivariate_visuals(df))
        sections.append(generate_multivariate_visuals(df))

    for section in sections:
        st.markdown(f"### 📌 {section['title']}")
        st.write(section['text'])
        if 'image' in section and section['image']:
            st.image(section['image'], use_container_width=True)

    st.subheader("🔍 Key Insights & Hypotheses")
    key_insights = generate_key_insights(df)
    st.write(key_insights)
    sections.append({"title": "Key Insights & Hypotheses", "text": key_insights})

    if st.button("📥 Generate & Download PDF Report"):
        pdf_file = save_pdf(sections)
        with open(pdf_file, "rb") as file:
            st.download_button(label="Download Report", data=file, file_name=pdf_file, mime="application/pdf")
        st.success("✅ Report generated successfully!")

    st.subheader("📝 Run Custom Query")
    query = st.text_area("Write your query (e.g., df.head(), df.describe())", "df.head()")
    if st.button("Execute Query"):
        try:
            result = eval(query, {"df": df, "pd": pd, "sns": sns, "plt": plt})
            st.write(result)
        except Exception as e:
            st.error(f"⚠️ Error executing query: {e}")

    st.success("🏁 AI-Enhanced Hypothesis & Report Generation Complete!")
