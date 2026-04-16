import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote

st.set_page_config(page_title="Maintenance prédictive 994F", layout="wide")

st.title("Maintenance prédictive - Flotte d'engins lourds")
st.markdown("Dashboard, score de risque, OT, planification et email automatique")

# -------------------------------------------------
# Fonctions utiles
# -------------------------------------------------
def lire_fichier(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file), None, None, None
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names

        feuille_principale = "Données_nettoyées" if "Données_nettoyées" in sheets else sheets[0]
        df_main = pd.read_excel(uploaded_file, sheet_name=feuille_principale)

        df_alertes = pd.read_excel(uploaded_file, sheet_name="Alertes_critiques") if "Alertes_critiques" in sheets else None
        df_aberrantes = pd.read_excel(uploaded_file, sheet_name="Valeurs_aberrantes") if "Valeurs_aberrantes" in sheets else None
        df_resume = pd.read_excel(uploaded_file, sheet_name="Résumé_paramètres") if "Résumé_paramètres" in sheets else None

        return df_main, df_alertes, df_aberrantes, df_resume


def nettoyer_df(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["Val_Min", "Val_Moy", "Val_Max", "Amplitude", "Heure_num", "Code"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Heure", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Aberrant" in df.columns:
        df["Aberrant"] = (
            df["Aberrant"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "oui", "vrai"])
        )
    else:
        df["Aberrant"] = False

    if "Alerte" not in df.columns:
        df["Alerte"] = "Normal"

    return df


def score_alerte(x):
    x = str(x).lower()
    if "critique" in x or "rouge" in x or "🔴" in x:
        return 2
    elif "avert" in x or "orange" in x or "🟡" in x:
        return 1
    return 0


def calcul_score_risque(df):
    df = df.copy()

    df["Alerte_num"] = df["Alerte"].apply(score_alerte)

    # Base simple et crédible
    score = (
        0.45 * df["Alerte_num"] +
        0.35 * df["Aberrant"].astype(int)
    )

    # Si Val_Moy existe, on ajoute une composante statistique
    if "Val_Moy" in df.columns:
        moyenne = df["Val_Moy"].mean()
        ecart = df["Val_Moy"].std()

        if pd.notna(ecart) and ecart != 0:
            z = ((df["Val_Moy"] - moyenne) / ecart).abs()
            z = z.fillna(0)
            z = np.where(z > 3, 1.0, np.where(z > 2, 0.6, np.where(z > 1, 0.3, 0.0)))
            score = score + 0.20 * z

    max_score = score.max() if score.max() != 0 else 1
    df["Score_Risque"] = (score / max_score).clip(0, 1)

    def label_risque(x):
        if x >= 0.80:
            return "Critique"
        elif x >= 0.50:
            return "Élevé"
        elif x >= 0.30:
            return "Modéré"
        return "Faible"

    df["Niveau_Risque"] = df["Score_Risque"].apply(label_risque)

    return df


def definir_priorite(score):
    if score >= 0.80:
        return "Urgente"
    elif score >= 0.50:
        return "Haute"
    elif score >= 0.30:
        return "Moyenne"
    return "Faible"


def definir_planification(score):
    if score >= 0.80:
        return "Aujourd’hui"
    elif score >= 0.50:
        return "Sous 24h"
    elif score >= 0.30:
        return "Sous 3 jours"
    return "Surveillance"


def action_recommandee(categorie, parametre):
    c = str(categorie).lower()
    p = str(parametre).lower()

    if "temp" in c or "temp" in p:
        return "Contrôler refroidissement, huile et échange thermique"
    if "pression" in c or "pression" in p:
        return "Inspecter circuit hydraulique, pompe et fuites"
    if "vibration" in c or "vibr" in p:
        return "Contrôler roulements, fixation et jeu mécanique"
    if "frein" in c or "frein" in p:
        return "Contrôler usure et efficacité du système de freinage"
    if "elec" in c or "elec" in p:
        return "Vérifier alimentation, câblage et continuité"
    return "Inspection détaillée du sous-système concerné"


def generer_ot(df):
    df = df.copy()
    df["Priorite_OT"] = df["Score_Risque"].apply(definir_priorite)
    df["Planification"] = df["Score_Risque"].apply(definir_planification)

    if "Categorie" not in df.columns:
        df["Categorie"] = "Non définie"
    if "Parametre" not in df.columns:
        df["Parametre"] = "Non défini"

    df["Action_recommandee"] = df.apply(
        lambda row: action_recommandee(row.get("Categorie", ""), row.get("Parametre", "")),
        axis=1
    )

    df["OT_ID"] = ["OT-" + str(i).zfill(4) for i in range(1, len(df) + 1)]
    df["Statut_OT"] = np.where(df["Score_Risque"] >= 0.30, "À lancer", "Surveillance")

    ot = df[df["Score_Risque"] >= 0.30].copy()
    ot = ot.sort_values("Score_Risque", ascending=False)

    return ot


# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("Importer votre fichier Excel ou CSV", type=["xlsx", "csv"])

if uploaded_file:
    df_main, df_alertes, df_aberrantes, df_resume = lire_fichier(uploaded_file)
    df_main = nettoyer_df(df_main)
    df_main = calcul_score_risque(df_main)

    # -------------------------
    # Sidebar filtres
    # -------------------------
    st.sidebar.header("Filtres")

    df_f = df_main.copy()

    if "Engin" in df_f.columns:
        engins = ["Tous"] + sorted(df_f["Engin"].dropna().astype(str).unique().tolist())
        engin_sel = st.sidebar.selectbox("Engin", engins)
        if engin_sel != "Tous":
            df_f = df_f[df_f["Engin"].astype(str) == engin_sel]

    if "Categorie" in df_f.columns:
        cats = ["Toutes"] + sorted(df_f["Categorie"].dropna().astype(str).unique().tolist())
        cat_sel = st.sidebar.selectbox("Catégorie", cats)
        if cat_sel != "Toutes":
            df_f = df_f[df_f["Categorie"].astype(str) == cat_sel]

    if "Parametre" in df_f.columns:
        params = ["Tous"] + sorted(df_f["Parametre"].dropna().astype(str).unique().tolist())
        param_sel = st.sidebar.selectbox("Paramètre", params)
        if param_sel != "Tous":
            df_f = df_f[df_f["Parametre"].astype(str) == param_sel]

    # recalcul après filtrage
    df_f = calcul_score_risque(df_f)
    ot = generer_ot(df_f)

    # -------------------------------------------------
    # Dashboard KPI
    # -------------------------------------------------
    st.subheader("Dashboard KPI")

    total_mesures = len(df_f)
    total_alertes_critiques = df_f["Alerte"].astype(str).str.contains("critique|rouge|🔴", case=False, na=False).sum()
    total_aberrantes = int(df_f["Aberrant"].sum()) if "Aberrant" in df_f.columns else 0
    score_moyen = round(df_f["Score_Risque"].mean() * 100, 2) if len(df_f) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mesures", f"{total_mesures:,}".replace(",", " "))
    c2.metric("Alertes critiques", int(total_alertes_critiques))
    c3.metric("Valeurs aberrantes", total_aberrantes)
    c4.metric("Score risque moyen", f"{score_moyen}%")

    # -------------------------------------------------
    # Onglets
    # -------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Aperçu base",
        "Prédiction / Risque",
        "Ordres de Travail",
        "Planification",
        "Email automatique"
    ])

    with tab1:
        st.subheader("Aperçu de la base de données")
        st.dataframe(df_f.head(50), use_container_width=True)

        if df_resume is not None:
            st.subheader("Résumé des paramètres")
            st.dataframe(df_resume, use_container_width=True)

        if df_alertes is not None:
            st.subheader("Alertes critiques")
            st.dataframe(df_alertes, use_container_width=True)

        if df_aberrantes is not None:
            st.subheader("Valeurs aberrantes")
            st.dataframe(df_aberrantes, use_container_width=True)

    with tab2:
        st.subheader("Prédiction du risque de panne")
        st.markdown("La prédiction est représentée ici par un **score de risque** basé sur les alertes, les valeurs aberrantes et les dérives statistiques.")

        cols_pred = [c for c in ["Engin", "Categorie", "Parametre", "Val_Min", "Val_Moy", "Val_Max", "Alerte", "Aberrant", "Score_Risque", "Niveau_Risque"] if c in df_f.columns]
        st.dataframe(
            df_f.sort_values("Score_Risque", ascending=False)[cols_pred].head(100),
            use_container_width=True
        )

        st.subheader("Répartition des niveaux de risque")
        repartition = df_f["Niveau_Risque"].value_counts()
        st.bar_chart(repartition)

    with tab3:
        st.subheader("Ordres de Travail automatiques")

        if len(ot) == 0:
            st.warning("Aucun OT à lancer selon le seuil actuel.")
        else:
            ot_cols = [c for c in [
                "OT_ID", "Engin", "Categorie", "Parametre", "Score_Risque",
                "Niveau_Risque", "Priorite_OT", "Action_recommandee", "Statut_OT"
            ] if c in ot.columns]

            st.dataframe(ot[ot_cols], use_container_width=True)

            csv_ot = ot.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les OT en CSV",
                data=csv_ot,
                file_name="OT_maintenance.csv",
                mime="text/csv"
            )

    with tab4:
        st.subheader("Planification de maintenance")

        if len(ot) == 0:
            st.warning("Aucune planification disponible.")
        else:
            plan_cols = [c for c in [
                "OT_ID", "Engin", "Categorie", "Parametre",
                "Priorite_OT", "Planification", "Action_recommandee"
            ] if c in ot.columns]

            st.dataframe(ot[plan_cols], use_container_width=True)

            st.subheader("Répartition des priorités")
            repart_priorite = ot["Priorite_OT"].value_counts()
            st.bar_chart(repart_priorite)

    with tab5:
        st.subheader("Email automatique")
        destinataire = "hourriyaamhaoul762@gmail.com"

        top_ot = ot.head(5).copy() if len(ot) > 0 else pd.DataFrame()

        if len(top_ot) > 0:
            lignes_ot = []
            for _, row in top_ot.iterrows():
                lignes_ot.append(
                    f"- {row.get('Engin', '')} | {row.get('Parametre', '')} | "
                    f"Risque={round(row.get('Score_Risque', 0)*100, 1)}% | "
                    f"Priorité={row.get('Priorite_OT', '')} | "
                    f"Planification={row.get('Planification', '')}"
                )
            resume_ot = "\n".join(lignes_ot)
        else:
            resume_ot = "Aucun ordre de travail prioritaire détecté."

        message = f"""
Bonjour,

Voici le rapport automatique de maintenance prédictive.

1. Synthèse :
- Nombre total de mesures : {total_mesures}
- Nombre d'alertes critiques : {int(total_alertes_critiques)}
- Nombre de valeurs aberrantes : {int(total_aberrantes)}
- Score moyen de risque : {score_moyen} %

2. Prédiction :
Le système a évalué le risque de défaillance à partir des données disponibles
(alertes, valeurs aberrantes, dérives des mesures et paramètres suivis).

3. Ordres de travail proposés :
{resume_ot}

4. Planification recommandée :
- Risque critique : intervention aujourd’hui
- Risque élevé : intervention sous 24h
- Risque modéré : intervention sous 3 jours
- Risque faible : surveillance

Cordialement,
Système intelligent de maintenance prédictive
"""

        st.text_area("Contenu de l'email", message, height=320)

        sujet = "Rapport Maintenance Prédictive - OT et Planification"
        gmail_link = f"https://mail.google.com/mail/?view=cm&to={destinataire}&su={quote(sujet)}&body={quote(message)}"

        st.markdown(f"[📩 Ouvrir Gmail avec le message prêt]({gmail_link})")

        st.success("Le système a généré automatiquement un rapport intelligent prêt à être envoyé.")

else:
    st.info("Importer votre fichier Excel ou CSV pour commencer.")
