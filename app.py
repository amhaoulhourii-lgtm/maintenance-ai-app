import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Maintenance prédictive 994F", layout="wide")

st.title("Maintenance prédictive - Chargeuse 994F")
st.markdown("Analyse des données, alertes, KPI et score de risque")

file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])

if file:
    xls = pd.ExcelFile(file)
    sheet_names = xls.sheet_names

    st.sidebar.header("Navigation")
    selected_sheet = st.sidebar.selectbox("Feuille principale", sheet_names)

    # Chargement des feuilles si elles existent
    df_main = pd.read_excel(file, sheet_name="Données_nettoyées") if "Données_nettoyées" in sheet_names else pd.read_excel(file, sheet_name=selected_sheet)
    df_resume = pd.read_excel(file, sheet_name="Résumé_paramètres") if "Résumé_paramètres" in sheet_names else None
    df_alertes = pd.read_excel(file, sheet_name="Alertes_critiques") if "Alertes_critiques" in sheet_names else None
    df_aberrantes = pd.read_excel(file, sheet_name="Valeurs_aberrantes") if "Valeurs_aberrantes" in sheet_names else None

    # Nettoyage noms colonnes
    df_main.columns = [str(c).strip() for c in df_main.columns]

    # Conversion numérique si colonnes présentes
    for col in ["Val_Min", "Val_Moy", "Val_Max"]:
        if col in df_main.columns:
            df_main[col] = pd.to_numeric(df_main[col], errors="coerce")

    # Conversion heure si présente
    if "Heure" in df_main.columns:
        df_main["Heure"] = pd.to_datetime(df_main["Heure"], errors="coerce")

    st.subheader("Aperçu des données")
    st.dataframe(df_main.head(20), use_container_width=True)

    # Sidebar filtres
    st.sidebar.header("Filtres")

    if "Engin" in df_main.columns:
        engins = ["Tous"] + sorted(df_main["Engin"].dropna().astype(str).unique().tolist())
        engin_sel = st.sidebar.selectbox("Engin", engins)
    else:
        engin_sel = "Tous"

    if "Categorie" in df_main.columns:
        categories = ["Toutes"] + sorted(df_main["Categorie"].dropna().astype(str).unique().tolist())
        cat_sel = st.sidebar.selectbox("Catégorie", categories)
    else:
        cat_sel = "Toutes"

    if "Parametre" in df_main.columns:
        params = ["Tous"] + sorted(df_main["Parametre"].dropna().astype(str).unique().tolist())
        param_sel = st.sidebar.selectbox("Paramètre", params)
    else:
        param_sel = "Tous"

    df_filt = df_main.copy()

    if engin_sel != "Tous" and "Engin" in df_filt.columns:
        df_filt = df_filt[df_filt["Engin"].astype(str) == engin_sel]

    if cat_sel != "Toutes" and "Categorie" in df_filt.columns:
        df_filt = df_filt[df_filt["Categorie"].astype(str) == cat_sel]

    if param_sel != "Tous" and "Parametre" in df_filt.columns:
        df_filt = df_filt[df_filt["Parametre"].astype(str) == param_sel]

    st.subheader("KPI")

    total_lignes = len(df_filt)

    nb_alertes = 0
    if "Alerte" in df_filt.columns:
        nb_alertes = df_filt["Alerte"].astype(str).str.contains("critique|alerte|rouge", case=False, na=False).sum()

    nb_aberrantes = 0
    if "Aberrant" in df_filt.columns:
        nb_aberrantes = df_filt["Aberrant"].astype(str).str.lower().isin(["true", "1", "oui", "vrai"]).sum()

    nb_parametres = df_filt["Parametre"].nunique() if "Parametre" in df_filt.columns else 0
    nb_categories = df_filt["Categorie"].nunique() if "Categorie" in df_filt.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes filtrées", f"{total_lignes:,}".replace(",", " "))
    c2.metric("Alertes détectées", nb_alertes)
    c3.metric("Valeurs aberrantes", nb_aberrantes)
    c4.metric("Paramètres suivis", nb_parametres)

    # Score de risque simple
    st.subheader("Score de risque")

    df_score = df_filt.copy()

    if "Val_Moy" in df_score.columns:
        mean_val = df_score["Val_Moy"].mean()
        std_val = df_score["Val_Moy"].std()

        if pd.notna(std_val) and std_val != 0:
            df_score["z_score"] = (df_score["Val_Moy"] - mean_val) / std_val
        else:
            df_score["z_score"] = 0

        def compute_risk(row):
            score = 0

            if "Aberrant" in row.index:
                val = str(row["Aberrant"]).lower()
                if val in ["true", "1", "oui", "vrai"]:
                    score += 50

            if "Alerte" in row.index:
                alert = str(row["Alerte"]).lower()
                if "critique" in alert or "rouge" in alert:
                    score += 40
                elif "alerte" in alert or "orange" in alert:
                    score += 25

            if "z_score" in row.index and pd.notna(row["z_score"]):
                if abs(row["z_score"]) > 3:
                    score += 30
                elif abs(row["z_score"]) > 2:
                    score += 15

            return min(score, 100)

        df_score["Score_Risque"] = df_score.apply(compute_risk, axis=1)

        def risk_label(x):
            if x >= 70:
                return "Critique"
            elif x >= 40:
                return "Élevé"
            elif x >= 20:
                return "Modéré"
            return "Faible"

        df_score["Niveau_Risque"] = df_score["Score_Risque"].apply(risk_label)

        col1, col2 = st.columns(2)
        col1.metric("Score risque moyen", round(df_score["Score_Risque"].mean(), 2))
        col2.metric("Nb risques critiques", (df_score["Niveau_Risque"] == "Critique").sum())

    # Tableau alertes critiques
    st.subheader("Alertes critiques")

    if df_alertes is not None:
        st.dataframe(df_alertes, use_container_width=True)
    elif "Alerte" in df_score.columns:
        crit = df_score[df_score["Alerte"].astype(str).str.contains("critique|rouge", case=False, na=False)]
        st.dataframe(crit, use_container_width=True)
    else:
        st.info("Aucune feuille d'alertes critiques trouvée.")

    # Tableau aberrantes
    st.subheader("Valeurs aberrantes")

    if df_aberrantes is not None:
        st.dataframe(df_aberrantes, use_container_width=True)
    elif "Aberrant" in df_score.columns:
        aberr = df_score[df_score["Aberrant"].astype(str).str.lower().isin(["true", "1", "oui", "vrai"])]
        st.dataframe(aberr, use_container_width=True)
    else:
        st.info("Aucune feuille de valeurs aberrantes trouvée.")

    # Répartition des alertes
    if "Alerte" in df_score.columns:
        st.subheader("Répartition des alertes")
        alert_counts = df_score["Alerte"].astype(str).value_counts()

        fig1, ax1 = plt.subplots()
        alert_counts.plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Type alerte")
        ax1.set_ylabel("Nombre")
        ax1.set_title("Distribution des alertes")
        st.pyplot(fig1)

    # Répartition du risque
    if "Niveau_Risque" in df_score.columns:
        st.subheader("Répartition du niveau de risque")
        risk_counts = df_score["Niveau_Risque"].value_counts()

        fig2, ax2 = plt.subplots()
        risk_counts.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Niveau de risque")
        ax2.set_ylabel("Nombre")
        ax2.set_title("Distribution du risque")
        st.pyplot(fig2)

    # Evolution temporelle
    if "Heure" in df_score.columns and "Val_Moy" in df_score.columns:
        st.subheader("Évolution temporelle de la valeur moyenne")

        df_time = df_score.dropna(subset=["Heure", "Val_Moy"]).sort_values("Heure")

        if len(df_time) > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(df_time["Heure"], df_time["Val_Moy"])
            ax3.set_xlabel("Heure")
            ax3.set_ylabel("Val_Moy")
            ax3.set_title("Évolution de Val_Moy")
            st.pyplot(fig3)

    # Résumé paramètres
    if df_resume is not None:
        st.subheader("Résumé des paramètres")
        st.dataframe(df_resume, use_container_width=True)

    # Export filtré
    st.subheader("Téléchargement des résultats")
    csv = df_score.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Télécharger les données filtrées en CSV",
        data=csv,
        file_name="resultats_maintenance_994F.csv",
        mime="text/csv",
    )

else:
    st.info("Téléversez un fichier Excel pour commencer.")
