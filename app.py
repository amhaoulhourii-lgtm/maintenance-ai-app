import streamlit as st
import pandas as pd

st.title("Maintenance des applications IA")

file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write(df.head())
