"""
🐟 Projet 8 — Classification de la qualité du poisson
Application Streamlit — Déploiement web
Basée sur le pipeline du notebook Fish_Classification_Notebook.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
from pathlib import Path

# ─── Configuration page ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 1rem; }

    .result-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 6px solid #2E7D32;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-species { font-size: 2rem; font-weight: 800; color: #1B5E20; }
    .result-emoji   { font-size: 3rem; }

    .metric-card {
        background: #E3F2FD;
        border-left: 4px solid #1565C0;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
    }
    .info-box {
        background: #FFF8E1;
        border-left: 4px solid #F57F17;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: #1565C0;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        width: 100%;
        padding: 0.6rem;
    }
    .stButton>button:hover { background: #0D47A1; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement des modèles ──────────────────────────────────────────────────
@st.cache_resource
def load_everything():
    base = Path(__file__).parent / "models"
    if not base.exists():
        base = Path("models")

    scaler = joblib.load(base / "scaler.pkl")
    le     = joblib.load(base / "label_encoder.pkl")

    models = {
        "🌲 Random Forest":       joblib.load(base / "model_rf.pkl"),
        "⚙️ SVM":                 joblib.load(base / "model_svm.pkl"),
        "📏 k-NN":                joblib.load(base / "model_knn.pkl"),
        "🌿 Arbre de décision":   joblib.load(base / "model_dt.pkl"),
    }

    with open(base / "meta.json") as f:
        meta = json.load(f)

    return scaler, le, models, meta

scaler, le, models, meta = load_everything()

FEATURES  = meta["features"]
CLASSES   = meta["classes"]
BEST_MODEL_NAME = meta["best_model"]
RESULTS_DF = pd.DataFrame(meta["results"]).sort_values("F1-score", ascending=False)

EMOJI = {
    "Bream":     "🐡", "Parkki":    "🐠", "Perch":     "🐟",
    "Pike":      "🦈", "Roach":     "🐡", "Smelt":     "✨",
    "Whitefish": "🫧",
}

MODEL_LABELS = {
    "k-NN":              "📏 k-NN",
    "Arbre de décision": "🌿 Arbre de décision",
    "Random Forest":     "🌲 Random Forest",
    "SVM":               "⚙️ SVM",
}
BEST_LABEL = MODEL_LABELS.get(BEST_MODEL_NAME, BEST_MODEL_NAME)


def predict(X_arr, model):
    scaled = scaler.transform(X_arr)
    labels = le.inverse_transform(model.predict(scaled))
    probas = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(scaled)
        probas = {cls: p[:, i] for i, cls in enumerate(le.classes_)}
    return labels, probas


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🐟 Fish Species Classifier</h1>
  <p>Projet 8 — Classification de la qualité du poisson · Fish Market Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Sélection du modèle")

    # Build options with ★ for best
    options = []
    for label in ["📏 k-NN", "🌿 Arbre de décision", "🌲 Random Forest", "⚙️ SVM"]:
        short = label.split(" ", 1)[1]           # strip emoji prefix
        is_best = (short == BEST_MODEL_NAME)
        options.append(label + (" ★" if is_best else ""))

    sel_raw = st.radio("Algorithme", options, index=next(
        i for i, o in enumerate(options) if "★" in o))

    # Strip the ★ suffix to get the key
    sel_key = sel_raw.rstrip(" ★")
    model_obj = models[sel_key]

    st.markdown("---")
    st.markdown("### 📊 Performances (jeu de test)")
    # Show perf for selected model
    short_name = sel_key.split(" ", 1)[1]   # "Random Forest" etc.
    row = next((r for r in meta["results"] if r["Modèle"] == short_name), None)
    if row:
        st.markdown(f"**Accuracy :** {row['Accuracy']:.1%}")
        st.markdown(f"**F1-score :** {row['F1-score']:.4f}")
        st.markdown(f"**CV Accuracy :** {row['CV Accuracy']:.1%}")
        params_str = meta["params"].get(short_name, "")
        if params_str:
            with st.expander("🔧 Meilleurs hyperparamètres"):
                st.code(params_str)

    st.markdown("---")
    st.markdown("### 🐟 Espèces prises en charge")
    for sp, em in EMOJI.items():
        st.markdown(f"{em} **{sp}**")

    st.markdown("---")
    st.markdown("### 🏆 Classement des modèles (F1-score)")
    for _, r in RESULTS_DF.iterrows():
        star = " ★" if r["Modèle"] == BEST_MODEL_NAME else ""
        st.markdown(f"- **{r['Modèle']}{star}** : {r['F1-score']:.4f}")

# ─── Onglets ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Test individuel", "📂 Test par fichier (CSV / Excel)", "📖 À propos du projet"])

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — TEST INDIVIDUEL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Saisie manuelle des caractéristiques d'un individu")

    col1, col2, col3 = st.columns(3)
    with col1:
        weight  = st.number_input("⚖️ Weight (g)",   0.0, 2000.0, 500.0, 1.0,
                                   help="Poids du poisson en grammes")
        length1 = st.number_input("📏 Length1 (cm)", 0.0, 80.0,   30.0, 0.1,
                                   help="Longueur verticale")
    with col2:
        length2 = st.number_input("📏 Length2 (cm)", 0.0, 80.0,   32.5, 0.1,
                                   help="Longueur diagonale")
        length3 = st.number_input("📏 Length3 (cm)", 0.0, 80.0,   35.0, 0.1,
                                   help="Longueur transversale")
    with col3:
        height  = st.number_input("📐 Height (cm)",  0.0, 40.0,   14.0, 0.1,
                                   help="Hauteur du poisson")
        width   = st.number_input("📐 Width (cm)",   0.0, 20.0,    5.0, 0.1,
                                   help="Largeur diagonale")

    st.markdown("")
    if st.button("🔍 Prédire l'espèce", key="btn_individuel"):
        arr = np.array([[weight, length1, length2, length3, height, width]])
        labels, probas = predict(arr, model_obj)
        species = labels[0]
        emoji   = EMOJI.get(species, "🐟")

        col_res, col_prob = st.columns([1, 2])
        with col_res:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-emoji">{emoji}</div>
              <div class="result-species">{species}</div>
              <div style="color:#388E3C;margin-top:0.4rem;font-size:0.9rem">
                Modèle : {sel_key}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_prob:
            if probas:
                st.markdown("**Probabilités par espèce**")
                for sp in sorted(probas, key=lambda s: probas[s][0], reverse=True):
                    p = float(probas[sp][0])
                    em = EMOJI.get(sp, "🐟")
                    st.markdown(f"{em} **{sp}**")
                    st.progress(p, text=f"{p:.1%}")
            else:
                st.info("Ce modèle ne fournit pas de probabilités (SVC sans `probability=True`).")

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — TEST PAR FICHIER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Import d'un fichier CSV ou Excel")

    st.markdown("""
    <div class="info-box">
    📋 <strong>Format attendu</strong> : colonnes <code>Weight</code>, <code>Length1</code>,
    <code>Length2</code>, <code>Length3</code>, <code>Height</code>, <code>Width</code>
    (dans n'importe quel ordre).<br>
    La colonne <code>Species</code> est <em>optionnelle</em> — si présente, une comparaison
    avec les vraies étiquettes sera effectuée.
    </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Template download
    tpl = pd.DataFrame({
        "Weight":  [500.0, 1000.0, 300.0],
        "Length1": [30.0,  40.0,   22.0],
        "Length2": [32.5,  43.0,   24.0],
        "Length3": [35.0,  46.5,   26.0],
        "Height":  [14.0,  16.0,   10.5],
        "Width":   [5.0,   6.5,    3.8],
    })
    st.download_button("📥 Télécharger un template CSV",
                       tpl.to_csv(index=False).encode("utf-8"),
                       "template_fish.csv", "text/csv")

    uploaded = st.file_uploader("📤 Importer votre fichier (CSV ou Excel)",
                                type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            df_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                    else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"❌ Impossible de lire le fichier : {e}")
            st.stop()

        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.error(f"❌ Colonnes manquantes : {missing}")
            st.stop()

        st.success(f"✅ **{len(df_in)} individus** chargés.")
        st.dataframe(df_in.head(5), use_container_width=True)

        if st.button("🚀 Lancer les prédictions", key="btn_fichier"):
            X_in = df_in[FEATURES].values
            labels, probas = predict(X_in, model_obj)

            df_out = df_in.copy()
            df_out["Espèce_prédite"] = labels
            df_out["Emoji"] = [EMOJI.get(s, "🐟") for s in labels]

            if probas:
                for sp in CLASSES:
                    df_out[f"Proba_{sp}"] = probas[sp].round(4)

            has_true = "Species" in df_out.columns
            if has_true:
                df_out["✔ Correct"] = df_out["Species"] == df_out["Espèce_prédite"]
                acc = df_out["✔ Correct"].mean()
                st.metric("🎯 Précision sur le fichier", f"{acc:.1%}")

            st.markdown("### 📋 Résultats")
            display_cols = FEATURES + ["Espèce_prédite", "Emoji"]
            if has_true:
                display_cols = ["Species"] + FEATURES + ["Espèce_prédite", "✔ Correct"]
            st.dataframe(df_out[display_cols], use_container_width=True)

            # Distribution
            st.markdown("### 📊 Distribution des espèces prédites")
            dist = df_out["Espèce_prédite"].value_counts().rename_axis("Espèce").reset_index(name="Nombre")
            st.bar_chart(dist.set_index("Espèce")["Nombre"])

            # Exports
            st.markdown("### 💾 Télécharger les résultats")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Exporter CSV",
                                   df_out.to_csv(index=False).encode("utf-8"),
                                   "predictions_fish.csv", "text/csv",
                                   use_container_width=True)
            with c2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df_out.to_excel(w, index=False, sheet_name="Prédictions")
                buf.seek(0)
                st.download_button("📥 Exporter Excel",
                                   buf.getvalue(),
                                   "predictions_fish.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 📌 Contexte
        Ce projet applique des algorithmes de **classification supervisée**
        pour prédire l'**espèce d'un poisson** à partir de ses mesures physiques.

        **Dataset :** Fish Market (159 observations, 7 espèces)  
        **Variables prédictives :** Weight, Length1, Length2, Length3, Height, Width  
        **Variable cible :** Species (7 classes)

        ### 🔬 Pipeline ML (conforme au notebook)
        1. **EDA** — distribution, histogrammes, boxplots, corrélation  
        2. **Détection des outliers** — méthode IQR  
        3. **Encodage** — `LabelEncoder` sur Species  
        4. **Split stratifié** 80/20 (`random_state=42`)  
        5. **Normalisation** — `StandardScaler` (fit sur train)  
        6. **GridSearchCV** — 5 folds, `StratifiedKFold`  
        7. **4 algorithmes** testés et comparés  
        """)

    with col_b:
        st.markdown("### 📊 Comparaison des modèles")
        display_df = RESULTS_DF[["Modèle","CV Accuracy","Accuracy","Precision","Recall","F1-score"]].copy()
        # Highlight best
        def highlight_best(row):
            return ["background-color: #C8E6C9" if row["Modèle"] == BEST_MODEL_NAME
                    else "" for _ in row]
        st.dataframe(
            display_df.style.apply(highlight_best, axis=1).format(
                {"CV Accuracy": "{:.1%}", "Accuracy": "{:.1%}",
                 "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"}
            ),
            use_container_width=True, hide_index=True
        )

        st.markdown(f"**🏆 Meilleur modèle : {BEST_MODEL_NAME}**")
        st.markdown("""
        ### 🛠️ Technologies
        Python · scikit-learn · Streamlit · pandas · joblib
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#90A4AE;font-size:0.85rem;">
      Projet 8 — Fish Market Classification · Pipeline conforme au notebook
    </div>""", unsafe_allow_html=True)
