import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PredictMaint Pro", layout="wide")

# ---------------------------
# CSS PRO
# ---------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.topbar {
    background: white;
    padding: 14px 24px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 14px;
}

.brand-title {
    font-size: 26px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 2px;
}

.brand-sub {
    color: #6b7280;
    font-size: 14px;
}

.machine-banner {
    background: linear-gradient(120deg, #f6c451 0%, #f0b429 100%);
    border-radius: 20px;
    padding: 26px;
    color: #1f2937;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
}

.machine-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.machine-sub {
    font-size: 15px;
    color: #374151;
}

.kpi-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 14px;
    min-height: 120px;
}

.kpi-title {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
}

.kpi-unit {
    font-size: 15px;
    color: #6b7280;
    font-weight: 500;
}

.kpi-target {
    margin-top: 10px;
    font-size: 13px;
    color: #6b7280;
}

.progress-wrap {
    background: #e5e7eb;
    border-radius: 999px;
    height: 9px;
    width: 100%;
    margin-top: 10px;
}

.progress-bar {
    background: #f59e0b;
    height: 9px;
    border-radius: 999px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
    margin: 18px 0 12px 0;
}

.subsystem-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 14px;
}

.subsystem-name {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
}

.subsystem-meta {
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 12px;
}

.subsystem-health {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
}

.health-good { color: #16a34a; }
.health-mid { color: #f59e0b; }
.health-bad { color: #dc2626; }

.alert-box {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 14px;
}

.alert-item {
    border-left: 5px solid #f59e0b;
    background: #fff7ed;
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.alert-item.info {
    border-left-color: #3b82f6;
    background: #eff6ff;
}

.alert-item.danger {
    border-left-color: #dc2626;
    background: #fef2f2;
}

.alert-title {
    font-weight: 700;
    color: #92400e;
    margin-bottom: 4px;
}

.alert-title.info {
    color: #1d4ed8;
}

.alert-title.danger {
    color: #991b1b;
}

.alert-meta {
    font-size: 13px;
    color: #6b7280;
}

.health-ring {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: conic-gradient(#f59e0b 0deg, #f59e0b 295deg, #e5e7eb 295deg 360deg);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 6px auto;
}

.health-ring-inner {
    width: 92px;
    height: 92px;
    border-radius: 50%;
    background: white;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.health-score {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    line-height: 1;
}

.health-label {
    font-size: 12px;
    color: #6b7280;
}

.small-tag {
    display: inline-block;
    background: #fff7ed;
    color: #b45309;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Données simulées pro
# ---------------------------
health_score = 82

kpis = [
    ("MTBF", "342", "h", 85),
    ("MTTR", "4.2", "h", 75),
    ("Disponibilité", "92.4", "%", 92),
    ("OEE", "78.6", "%", 79),
    ("Efficacité carburant", "4.8", "t/L", 82),
    ("Coût / heure", "285", "$/h", 88),
]

subsystems = [
    ("Moteur Diesel", 8, 88, 127),
    ("Système Hydraulique", 7, 75, 43),
    ("Transmission", 4, 91, 312),
    ("Système de Refroidissement", 4, 84, 210),
    ("Système électrique", 3, 93, 540),
    ("Châssis et structure", 6, 78, 680),
    ("Système de Freinage", 4, 86, 178),
]

alerts = [
    ("ATTENTION", "ΔP filtre hydraulique élevé - Colmatage probable.", "Système Hydraulique · il y a 30 min", "warning"),
    ("ATTENTION", "Niveau huile hydraulique bas - Vérifier les fuites et compléter le niveau.", "Système Hydraulique · il y a 1 h", "warning"),
    ("ATTENTION", "Pression pneu arrière droit basse - Vérifier l'état du pneu.", "Châssis et structure · il y a 2 h", "warning"),
    ("INFO", "Entretien préventif transmission dans 312 heures.", "Transmission · il y a 5 min", "info"),
]

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="topbar">
    <div class="brand-title">PredictMaint Pro</div>
    <div class="brand-sub">Maintenance Prédictive & Conditionnelle — CBM</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Bannière machine
# ---------------------------
col_banner, col_ring = st.columns([5, 1.2])

with col_banner:
    st.markdown("""
    <div class="machine-banner">
        <div class="small-tag">CAT 994F</div>
        <div class="small-tag">36 capteurs</div>
        <div class="small-tag">3 alertes</div>
        <div class="small-tag">7 sous-systèmes</div>
        <div class="machine-title">CAT 994F</div>
        <div class="machine-sub">S/N : AXJ00542 · Zone d'extraction Nord · Niveau 3</div>
    </div>
    """, unsafe_allow_html=True)

with col_ring:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:center;">
        <div class="health-ring">
            <div class="health-ring-inner">
                <div class="health-score">{health_score}</div>
                <div class="health-label">/ 100</div>
            </div>
        </div>
        <div class="brand-sub" style="text-align:center;">Santé globale</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# KPI row
# ---------------------------
kpi_cols = st.columns(6)

for i, (title, value, unit, progress) in enumerate(kpis):
    with kpi_cols[i]:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
            <div class="progress-wrap">
                <div class="progress-bar" style="width:{progress}%;"></div>
            </div>
            <div class="kpi-target">Cible : {progress}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Sous-systemes + alertes
# ---------------------------
left, right = st.columns([3.4, 1.1])

with left:
    st.markdown('<div class="section-title">Sous-systèmes</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (name, capteurs, sante, entretien) in enumerate(subsystems):
        color_class = "health-good" if sante >= 85 else "health-mid" if sante >= 70 else "health-bad"
        progress_color = "#16a34a" if sante >= 85 else "#f59e0b" if sante >= 70 else "#dc2626"

        with cols[idx % 3]:
            st.markdown(f"""
            <div class="subsystem-card">
                <div class="subsystem-name">{name}</div>
                <div class="subsystem-meta">{capteurs} capteurs</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="subsystem-meta">Santé</div>
                    <div class="subsystem-health {color_class}">{sante} %</div>
                </div>
                <div class="progress-wrap">
                    <div class="progress-bar" style="width:{sante}%; background:{progress_color};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <div class="subsystem-meta">Entretien prochain</div>
                    <div class="subsystem-meta"><b>{entretien} h</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Alertes actives</div>', unsafe_allow_html=True)
    st.markdown('<div class="alert-box">', unsafe_allow_html=True)

    for title, text, meta, kind in alerts:
        if kind == "info":
            st.markdown(f"""
            <div class="alert-item info">
                <div class="alert-title info">{title}</div>
                <div>{text}</div>
                <div class="alert-meta">{meta}</div>
            </div>
            """, unsafe_allow_html=True)
        elif kind == "danger":
            st.markdown(f"""
            <div class="alert-item danger">
                <div class="alert-title danger">{title}</div>
                <div>{text}</div>
                <div class="alert-meta">{meta}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-item">
                <div class="alert-title">{title}</div>
                <div>{text}</div>
                <div class="alert-meta">{meta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("Import de la base de données")
uploaded_file = st.file_uploader("Téléverser votre fichier Excel", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Choisir la feuille", xls.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)

    st.dataframe(df.head(20), use_container_width=True)
