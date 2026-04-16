import streamlit as st
import pandas as pd

st.title("Application Maintenance IA")

file = st.file_uploader("Upload ton fichier Excel", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write(df.head())
