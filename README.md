# 🐟 Fish Species Classifier — Projet 8

Application Streamlit déployable en ligne, basée sur le pipeline du notebook `Fish_Classification_Notebook.ipynb`.

## 📁 Structure du projet

```
prpjet-ML_C34639/
├── app.py                  ← Application Streamlit principale
├── requirements.txt        ← Dépendances Python
├── README.md               ← Ce fichier
├── Classification de qualité du poisson(C34639).ipynb  ← Notebook jupyter
├── Fish.csv                ← Dataset original
├── scaler.pkl          ← StandardScaler (fit sur X_train)
├── label_encoder.pkl   ← LabelEncoder des espèces
├── model_knn.pkl       ← Meilleur k-NN (GridSearchCV)
├── model_dt.pkl        ← Meilleur Arbre de décision (GridSearchCV)
├── model_rf.pkl        ← Meilleur Random Forest (GridSearchCV)
├── model_svm.pkl       ← Meilleur SVM (GridSearchCV)
├── best_model.pkl      ← Meilleur modèle global
└── meta.json           ← Métriques et hyperparamètres
```

## 🚀 Déploiement sur Streamlit Community Cloud (GRATUIT)

### Étapes

1. Aller sur **[share.streamlit.io](https://share.streamlit.io)**

2. Se connecter avec votre compte GitHub

3. Cliquer **"New app"** → sélectionner votre dépôt

4. **Main file path** : `app.py`

5. Cliquer **"Deploy"** → l'app sera disponible en ~2 minutes


## 💻 Lancement en local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

Ouvre automatiquement sur `http://localhost:8501`

## 🎯 Fonctionnalités

### Test individuel
- Saisie manuelle des 6 mesures biométriques
- Prédiction instantanée de l'espèce
- Affichage des probabilités par classe (barre de progression)
- Choix du modèle parmi les 4 algorithmes entraînés

### Test par fichier
- Import d'un fichier CSV ou Excel
- Prédictions en masse sur tous les individus
- Comparaison avec les vraies espèces si la colonne `Species` est présente
- Export des résultats en CSV ou Excel

### Sidebar
- Sélection du modèle (k-NN, Arbre de décision, Random Forest, SVM)
- Meilleur modèle identifié par ★
- Performances (Accuracy, F1-score, CV Accuracy)
- Hyperparamètres optimaux (GridSearchCV)

## 📊 Pipeline ML (conforme au notebook)

| Étape | Détail |
|-------|--------|
| EDA | Distribution, histogrammes, boxplots, corrélation |
| Outliers | Méthode IQR |
| Encodage | `LabelEncoder` sur `Species` |
| Split | 80/20 stratifié, `random_state=42` |
| Normalisation | `StandardScaler` (fit sur X_train uniquement) |
| Tuning | `GridSearchCV`, 5 folds, `StratifiedKFold` |
| Modèles | k-NN, Arbre de décision, Random Forest, SVM |
