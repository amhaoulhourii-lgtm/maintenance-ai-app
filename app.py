import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Maintenance prédictive 994F", layout="wide")

st.title("Application de maintenance prédictive")
st.markdown("Dashboard, prédiction des pannes, OT et planification maintenance")

# --------------------------------------------------
# Fonctions
# --------------------------------------------------
@st.cache_data
def load_excel(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names
    data = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in sheets}
    return data, sheets


def clean_df(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["Val_Min", "Val_Moy", "Val_Max", "Amplitude", "Heure_num", "Code"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Heure", "Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "Aberrant" in df.columns:
        df["Aberrant"] = df["Aberrant"].astype(str).str.lower().isin(["true", "1", "oui", "vrai"])

    if "Alerte" in df.columns:
        df["Alerte"] = df["Alerte"].astype(str)

    return df


def alert_to_num(x):
    x = str(x).lower()
    if "critique" in x or "rouge" in x or "🔴" in x:
        return 2
    elif "avert" in x or "orange" in x or "🟡" in x:
        return 1
    return 0


def build_model_df(df, horizon_hours=6):
    needed = ["Engin", "Parametre", "Heure"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    d = df.copy()
    d["Alerte_num"] = d["Alerte"].apply(alert_to_num) if "Alerte" in d.columns else 0
    d["Event_Critique"] = ((d["Alerte_num"] >= 2) | (d["Aberrant"] == True)).astype(int)

    d = d.sort_values(["Engin", "Parametre", "Heure"]).reset_index(drop=True)

    grp = ["Engin", "Parametre"]

    for col in ["Val_Min", "Val_Moy", "Val_Max", "Amplitude"]:
        if col in d.columns:
            d[f"{col}_lag1"] = d.groupby(grp)[col].shift(1)
            d[f"{col}_lag2"] = d.groupby(grp)[col].shift(2)
            d[f"{col}_roll3"] = d.groupby(grp)[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
            d[f"{col}_roll6"] = d.groupby(grp)[col].transform(lambda x: x.rolling(6, min_periods=1).mean())

    out = []
    for _, g in d.groupby(grp):
        g = g.sort_values("Heure").copy()
        target = np.zeros(len(g), dtype=int)

        for i in range(len(g)):
            t = g.iloc[i]["Heure"]
            if pd.isna(t):
                target[i] = 0
                continue
            t2 = t + pd.Timedelta(hours=horizon_hours)
            win = g[(g["Heure"] > t) & (g["Heure"] <= t2)]
            target[i] = 1 if (len(win) > 0 and win["Event_Critique"].max() == 1) else 0

        g["Panne_Future"] = target
        out.append(g)

    model_df = pd.concat(out, ignore_index=True)

    model_df["hour"] = model_df["Heure"].dt.hour if "Heure" in model_df.columns else 0
    model_df["day"] = model_df["Heure"].dt.day if "Heure" in model_df.columns else 0
    model_df["month"] = model_df["Heure"].dt.month if "Heure" in model_df.columns else 0

    return model_df


def train_predict(model_df):
    features = [
        "Val_Min", "Val_Moy", "Val_Max", "Amplitude",
        "Val_Min_lag1", "Val_Min_lag2", "Val_Min_roll3", "Val_Min_roll6",
        "Val_Moy_lag1", "Val_Moy_lag2", "Val_Moy_roll3", "Val_Moy_roll6",
        "Val_Max_lag1", "Val_Max_lag2", "Val_Max_roll3", "Val_Max_roll6",
        "Amplitude_lag1", "Amplitude_lag2", "Amplitude_roll3", "Amplitude_roll6",
        "Alerte_num", "Heure_num", "Code", "hour", "day", "month"
    ]
    features = [c for c in features if c in model_df.columns]

    X_num = model_df[features].copy()

    cat_cols = [c for c in ["Engin", "Categorie", "Shift", "Param_court"] if c in model_df.columns]
    if cat_cols:
        X_cat = pd.get_dummies(model_df[cat_cols].astype(str), drop_first=False)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = model_df["Panne_Future"]

    if y.nunique() < 2:
        return None, None, None, None, "Une seule classe trouvée dans la cible. Utilise plus de données."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred)
    }

    full_prob = model.predict_proba(X)[:, 1]
    scored = model_df.copy()
    scored["Prob_Panne"] = full_prob
    scored["Prediction_Panne"] = (scored["Prob_Panne"] >= 0.5).astype(int)

    feat_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, scored, metrics, feat_imp, None


def make_ot_plan(scored_df):
    d = scored_df.copy()

    def priority(p):
        if p >= 0.80:
            return "Urgente"
        elif p >= 0.50:
            return "Haute"
        elif p >= 0.30:
            return "Planifiée"
        return "Surveillance"

    def action(cat, param):
        cat = str(cat).lower()
        param = str(param).lower()

        if "temp" in cat or "temp" in param:
            return "Contrôler refroidissement, huile, échange thermique"
        if "pression" in cat or "pression" in param:
            return "Inspecter circuit hydraulique, pompe, fuites"
        if "vibration" in cat or "vibr" in param:
            return "Contrôler roulements, fixations, jeu mécanique"
        return "Inspection détaillée du composant"

    def planning(p):
        now = pd.Timestamp.now().floor("H")
        if p >= 0.80:
            return now
        elif p >= 0.50:
            return now + pd.Timedelta(hours=24)
        elif p >= 0.30:
            return now + pd.Timedelta(days=3)
        return now + pd.Timedelta(days=7)

    d["Priorite_OT"] = d["Prob_Panne"].apply(priority)
    d["Action_recommandee"] = d.apply(
        lambda r: action(r["Categorie"] if "Categorie" in d.columns else "", r["Parametre"] if "Parametre" in d.columns else ""),
        axis=1
    )
    d["Date_planifiee"] = d["Prob_Panne"].apply(planning)
    d["OT_ID"] = ["OT-" + str(i).zfill(4) for i in range(1, len(d) + 1)]
    d["Statut_OT"] = np.where(d["Prob_Panne"] >= 0.30, "À lancer", "Surveillance")

    return d


# --------------------------------------------------
# Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])

if uploaded_file:
    data, sheets = load_excel(uploaded_file)

    main_sheet = "Données_nettoyées" if "Données_nettoyées" in sheets else sheets[0]
    df = clean_df(data[main_sheet])

    df_alert = data["Alertes_critiques"] if "Alertes_critiques" in sheets else None
    df_ab = data["Valeurs_aberrantes"] if "Valeurs_aberrantes" in sheets else None
    df_res = data["Résumé_paramètres"] if "Résumé_paramètres" in sheets else None

    # Sidebar
    st.sidebar.header("Filtres")

    if "Engin" in df.columns:
        engins = ["Tous"] + sorted(df["Engin"].dropna().astype(str).unique().tolist())
        engin_sel = st.sidebar.selectbox("Engin", engins)
    else:
        engin_sel = "Tous"

    if "Categorie" in df.columns:
        cats = ["Toutes"] + sorted(df["Categorie"].dropna().astype(str).unique().tolist())
        cat_sel = st.sidebar.selectbox("Catégorie", cats)
    else:
        cat_sel = "Toutes"

    if "Parametre" in df.columns:
        params = ["Tous"] + sorted(df["Parametre"].dropna().astype(str).unique().tolist())
        param_sel = st.sidebar.selectbox("Paramètre", params)
    else:
        param_sel = "Tous"

    horizon = st.sidebar.slider("Horizon prédiction (heures)", 1, 24, 6)

    df_f = df.copy()
    if engin_sel != "Tous" and "Engin" in df_f.columns:
        df_f = df_f[df_f["Engin"].astype(str) == engin_sel]
    if cat_sel != "Toutes" and "Categorie" in df_f.columns:
        df_f = df_f[df_f["Categorie"].astype(str) == cat_sel]
    if param_sel != "Tous" and "Parametre" in df_f.columns:
        df_f = df_f[df_f["Parametre"].astype(str) == param_sel]

    # --------------------------------------------------
    # Tabs
    # --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard",
        "Prédiction",
        "OT",
        "Planification"
    ])

    # ---------------- Dashboard ----------------
    with tab1:
        st.subheader("Dashboard maintenance")

        nb_lignes = len(df_f)
        nb_engins = df_f["Engin"].nunique() if "Engin" in df_f.columns else 0
        nb_params = df_f["Parametre"].nunique() if "Parametre" in df_f.columns else 0
        nb_aberr = int(df_f["Aberrant"].sum()) if "Aberrant" in df_f.columns else 0
        nb_crit = df_f["Alerte"].astype(str).str.contains("critique|🔴|rouge", case=False, na=False).sum() if "Alerte" in df_f.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mesures", f"{nb_lignes:,}".replace(",", " "))
        c2.metric("Engins", nb_engins)
        c3.metric("Alertes critiques", int(nb_crit))
        c4.metric("Valeurs aberrantes", nb_aberr)

        c5, c6 = st.columns(2)
        c5.metric("Paramètres", nb_params)
        c6.metric("Taux d'alerte", f"{(nb_crit / nb_lignes * 100):.2f}%" if nb_lignes > 0 else "0%")

        st.subheader("Aperçu des données")
        st.dataframe(df_f.head(20), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            if "Alerte" in df_f.columns:
                fig, ax = plt.subplots()
                df_f["Alerte"].astype(str).value_counts().plot(kind="bar", ax=ax)
                ax.set_title("Répartition des alertes")
                ax.set_xlabel("Alerte")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)

        with col2:
            if "Categorie" in df_f.columns:
                fig, ax = plt.subplots()
                df_f["Categorie"].astype(str).value_counts().head(10).plot(kind="bar", ax=ax)
                ax.set_title("Top catégories")
                ax.set_xlabel("Catégorie")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)

        if "Heure" in df_f.columns and "Val_Moy" in df_f.columns:
            df_time = df_f.dropna(subset=["Heure", "Val_Moy"]).sort_values("Heure")
            if len(df_time) > 0:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_time["Heure"], df_time["Val_Moy"])
                ax.set_title("Évolution temporelle de Val_Moy")
                ax.set_xlabel("Temps")
                ax.set_ylabel("Val_Moy")
                st.pyplot(fig)

        st.subheader("Tables métier")
        t1, t2, t3 = st.tabs(["Alertes critiques", "Valeurs aberrantes", "Résumé"])
        with t1:
            if df_alert is not None:
                st.dataframe(df_alert, use_container_width=True)
        with t2:
            if df_ab is not None:
                st.dataframe(df_ab, use_container_width=True)
        with t3:
            if df_res is not None:
                st.dataframe(df_res, use_container_width=True)

    # ---------------- Prediction ----------------
    with tab2:
        st.subheader("Prédiction du risque de panne")

        model_df = build_model_df(df_f, horizon_hours=horizon)

        if model_df.empty:
            st.error("Colonnes nécessaires manquantes : Engin, Parametre, Heure.")
        else:
            st.write("Dataset de modélisation")
            st.dataframe(model_df.head(20), use_container_width=True)

            st.metric("Taux cible panne future", f"{model_df['Panne_Future'].mean() * 100:.2f}%")

            model, scored_df, metrics, feat_imp, err = train_predict(model_df)

            if err:
                st.warning(err)
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                m2.metric("Précision", f"{metrics['precision']:.3f}")
                m3.metric("Recall", f"{metrics['recall']:.3f}")

                st.subheader("Matrice de confusion")
                cm_df = pd.DataFrame(
                    metrics["cm"],
                    index=["Réel 0", "Réel 1"],
                    columns=["Prédit 0", "Prédit 1"]
                )
                st.dataframe(cm_df, use_container_width=True)

                st.subheader("Variables importantes")
                st.dataframe(feat_imp.head(15), use_container_width=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                feat_imp.head(10).sort_values("Importance").plot(
                    kind="barh", x="Feature", y="Importance", ax=ax
                )
                ax.set_title("Top 10 variables importantes")
                st.pyplot(fig)

                st.subheader("Top risques")
                top_risk = scored_df.sort_values("Prob_Panne", ascending=False).head(50)
                cols = [c for c in [
                    "Engin", "Parametre", "Heure", "Categorie", "Val_Min", "Val_Moy",
                    "Val_Max", "Amplitude", "Alerte", "Aberrant", "Prob_Panne"
                ] if c in top_risk.columns]
                st.dataframe(top_risk[cols], use_container_width=True)

                st.session_state["scored_df"] = scored_df

    # ---------------- OT ----------------
    with tab3:
        st.subheader("Génération des OT")

        if "scored_df" not in st.session_state:
            st.info("Lance d'abord la prédiction.")
        else:
            scored_df = st.session_state["scored_df"]
            ot_df = make_ot_plan(scored_df)
            ot_df = ot_df[ot_df["Prob_Panne"] >= 0.30].sort_values("Prob_Panne", ascending=False)

            ot_cols = [c for c in [
                "OT_ID", "Engin", "Parametre", "Heure", "Categorie",
                "Prob_Panne", "Priorite_OT", "Action_recommandee", "Statut_OT"
            ] if c in ot_df.columns]

            st.dataframe(ot_df[ot_cols], use_container_width=True)

            csv_ot = ot_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les OT",
                data=csv_ot,
                file_name="OT_maintenance.csv",
                mime="text/csv"
            )

            st.session_state["ot_df"] = ot_df

    # ---------------- Planning ----------------
    with tab4:
        st.subheader("Planification maintenance")

        if "ot_df" not in st.session_state:
            st.info("Génère d'abord les OT.")
        else:
            plan_df = st.session_state["ot_df"].copy()

            plan_cols = [c for c in [
                "OT_ID", "Engin", "Parametre", "Categorie",
                "Priorite_OT", "Date_planifiee", "Action_recommandee"
            ] if c in plan_df.columns]

            st.dataframe(plan_df[plan_cols], use_container_width=True)

            if "Priorite_OT" in plan_df.columns:
                fig, ax = plt.subplots()
                plan_df["Priorite_OT"].value_counts().plot(kind="bar", ax=ax)
                ax.set_title("Répartition des priorités OT")
                ax.set_xlabel("Priorité")
                ax.set_ylabel("Nombre")
                st.pyplot(fig)

            csv_plan = plan_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger le planning",
                data=csv_plan,
                file_name="planning_maintenance.csv",
                mime="text/csv"
            )

else:
    st.info("Téléversez un fichier Excel pour commencer.")
