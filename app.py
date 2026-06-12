"""
🐟 Fish Species Classifier — Application Streamlit
Compatible avec les fichiers placés à la racine du dépôt GitHub :
- best_model.pkl
- scaler.pkl
- label_encoder.pkl
- Fish.csv
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
)

FEATURES = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]

EMOJI = {
    "Bream": "🐡",
    "Parkki": "🐠",
    "Perch": "🐟",
    "Pike": "🦈",
    "Roach": "🐡",
    "Smelt": "✨",
    "Whitefish": "🫧",
}

@st.cache_resource
def load_everything():
    base = Path(__file__).parent

    model_path = base / "best_model.pkl"
    scaler_path = base / "scaler.pkl"
    encoder_path = base / "label_encoder.pkl"

    missing = [
        str(p.name)
        for p in [model_path, scaler_path, encoder_path]
        if not p.exists()
    ]

    if missing:
        st.error(
            "❌ Fichiers manquants : "
            + ", ".join(missing)
            + ". Vérifie qu'ils sont dans le même dossier que app.py sur GitHub."
        )
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    return model, scaler, label_encoder


@st.cache_data
def load_data():
    path = Path(__file__).parent / "Fish.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


model, scaler, label_encoder = load_everything()
df = load_data()

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);
    color:white;padding:1.5rem;border-radius:12px;text-align:center;">
        <h1>🐟 Fish Species Classifier</h1>
        <p>Prédiction de l'espèce d'un poisson à partir de ses caractéristiques morphologiques</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["🔮 Prédiction individuelle", "📂 Prédiction par fichier", "📊 Dataset", "ℹ️ À propos"],
)

def predict_array(X):
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    labels = label_encoder.inverse_transform(pred)

    probas = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)

    return labels, probas


if page == "🔮 Prédiction individuelle":
    st.header("🔮 Prédiction individuelle")

    col1, col2, col3 = st.columns(3)

    with col1:
        weight = st.number_input("Weight (g)", min_value=0.0, value=500.0)
        length1 = st.number_input("Length1 (cm)", min_value=0.0, value=30.0)

    with col2:
        length2 = st.number_input("Length2 (cm)", min_value=0.0, value=32.5)
        length3 = st.number_input("Length3 (cm)", min_value=0.0, value=35.0)

    with col3:
        height = st.number_input("Height (cm)", min_value=0.0, value=14.0)
        width = st.number_input("Width (cm)", min_value=0.0, value=5.0)

    if st.button("🔍 Prédire l'espèce", type="primary"):
        X = np.array([[weight, length1, length2, length3, height, width]])
        labels, probas = predict_array(X)
        species = labels[0]

        st.success(f"Espèce prédite : {EMOJI.get(species, '🐟')} **{species}**")

        if probas is not None:
            st.subheader("Probabilités par espèce")
            probabilities = pd.DataFrame({
                "Espèce": label_encoder.classes_,
                "Probabilité": probas[0]
            }).sort_values("Probabilité", ascending=False)

            probabilities["Probabilité"] = probabilities["Probabilité"].map(lambda x: f"{x:.2%}")
            st.dataframe(probabilities, use_container_width=True, hide_index=True)

elif page == "📂 Prédiction par fichier":
    st.header("📂 Prédiction par fichier")

    st.info(f"Le fichier doit contenir les colonnes : {', '.join(FEATURES)}")

    template = pd.DataFrame({
        "Weight": [500.0, 1200.0, 9.7],
        "Length1": [30.0, 56.0, 9.3],
        "Length2": [32.5, 58.0, 9.8],
        "Length3": [35.0, 65.0, 11.0],
        "Height": [14.0, 9.0, 2.1],
        "Width": [5.0, 8.0, 1.3],
    })

    st.download_button(
        "📥 Télécharger un template CSV",
        template.to_csv(index=False).encode("utf-8"),
        "template_fish.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            data = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            st.stop()

        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()

        X = data[FEATURES].values
        labels, probas = predict_array(X)

        results = data.copy()
        results["Espèce_prédite"] = labels

        if probas is not None:
            results["Confiance_%"] = (probas.max(axis=1) * 100).round(2)

        st.subheader("Résultats")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "📥 Télécharger les résultats CSV",
            results.to_csv(index=False).encode("utf-8"),
            "predictions_fish.csv",
            "text/csv",
        )

elif page == "📊 Dataset":
    st.header("📊 Dataset")

    if df is None:
        st.warning("Fish.csv n'a pas été trouvé dans le dépôt.")
    else:
        st.write(f"Nombre d'observations : **{len(df)}**")
        st.write(f"Nombre d'espèces : **{df['Species'].nunique()}**")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Répartition des espèces")
        st.bar_chart(df["Species"].value_counts())

else:
    st.header("ℹ️ À propos")
    st.write(
        """
        Cette application utilise un modèle de Machine Learning entraîné sur le dataset Fish Market.
        Elle prédit l'espèce d'un poisson à partir de six caractéristiques :
        Weight, Length1, Length2, Length3, Height et Width.
        """
    )
