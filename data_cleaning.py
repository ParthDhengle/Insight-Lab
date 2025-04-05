import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("🛠 Data Cleaning")

    # Ensure dataset is available
    if 'dataframe' not in st.session_state:
        st.warning("⚠️ No dataset found. Please upload a CSV file from the Home page.")
        return

    df = st.session_state['dataframe']

    st.subheader("📂 Current Data Preview")
    st.dataframe(df.head())

    # Step 1: Handle Missing Values
    if st.button("🧹 Handle Missing Values"):
        df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric columns with median
        df.fillna(df.mode(), inplace=True)  # Fill categorical columns with 'Unknown'
        st.session_state['dataframe'] = df  # Update session state
        st.success("✅ Missing values handled!")
        st.dataframe(df.head())

    # Step 2: Remove Duplicates
    if st.button("🗑 Remove Duplicates"):
        before = len(df)
        df.drop_duplicates(inplace=True)
        after = len(df)
        st.session_state['dataframe'] = df  # Update session state
        st.success(f"✅ Removed {before - after} duplicate rows.")
        st.dataframe(df.head())

    # Step 3: Convert Data Types
    if st.button("🔄 Convert Data Types"):
        for col in df.columns:
            if df[col].dtype == 'object':  # Convert categorical columns to category type
                df[col] = df[col].astype('category')
            elif df[col].dtype == 'float64':  # Convert floats that look like integers to int
                if all(df[col].dropna().apply(float.is_integer)):
                    df[col] = df[col].astype('int64')
        st.session_state['dataframe'] = df  # Update session state
        st.success("✅ Data types corrected!")
        st.dataframe(df.dtypes)

    # Step 4: Handle Outliers (Using IQR)
    if st.button("📉 Handle Outliers"):
        before = len(df)
        
        # Select only numeric columns for outlier removal
        numeric_cols = df.select_dtypes(include=['number']).columns
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        # Filter only numeric columns
        filter_mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

        # Apply filtering to the full dataframe
        df = df[filter_mask]

        after = len(df)
        st.session_state['dataframe'] = df  # Update session state
        st.success(f"✅ Removed {before - after} rows containing outliers.")
        st.dataframe(df.head())


    # Step 6: Save Cleaned Data
    if st.button("💾 Save Cleaned Data"):
        st.session_state['dataframe'] = df.copy() 
        st.success("✅ Cleaned data saved! You can now proceed to 'After Cleaning'.")

    # Custom Query Section
    st.subheader("🔍 Run Your Own Query")
    st.write("Enter a **pandas query** (e.g., `df[df['column_name'] > 100]`) and run it on the dataset.")

    user_query = st.text_area("Enter your pandas command here (use `df` as your dataset):")

    if st.button("▶ Run Query"):
        try:
            # Execute the user's query safely
            result = eval(user_query, {'df': df, 'pd': pd, 'np': np})
            st.write("✅ **Query Result:**")
            st.dataframe(result)

        except Exception as e:
            st.error(f"❌ Error in query execution: {e}")
