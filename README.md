#  Prédiction Immobilière Paris — Segmentation K-Means & Random Forest

> Application complète de data science pour analyser le marché immobilier parisien : segmentation automatique des biens en 5 clusters avec K-Means et prédiction de prix avec Random Forest (R² = 0.989), le tout via un dashboard Streamlit interactif avec carte Folium.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-77B829?style=flat-square)

---

##  Résultats clés

| Métrique | Valeur |
|----------|--------|
| **R²** | **0.989** (98.9% de variance expliquée) |
| **MAE** | 47 732 € |
| **RMSE** | 68 500 € |
| **Précision à ±10%** | 78% |
| **Clusters identifiés** | 5 segments de marché |
| **Biens analysés** | 2 000 (calibrés sur données DVF 2024) |

---

##  Pipeline

```
Collecte (DVF 2024) → Nettoyage (IQR, imputation) → Segmentation (K-Means, 5 clusters)
    → Prédiction (Random Forest, 100 arbres) → Dashboard interactif (Streamlit + Folium)
```

---

##  Segments de marché identifiés

| Cluster | Type | Biens | Prix moyen | Prix/m² | Profil |
|---------|------|-------|------------|---------|--------|
| 0 | Économique | 295 | 1.08 M€ | 16 553 € | Primo-accédants |
| 1 | Moyen | 837 | 1.68 M€ | 11 579 € | Familles |
| 2 | Haut de gamme | 60 | 763 k€ | 28 882 € | Investisseurs |
| 3 | Luxe | 160 | 817 k€ | 20 996 € | Haut revenu |
| 4 | Exceptionnel | 648 | 1.56 M€ | 13 534 € | Cadres, expatriés |

---

##  Fonctionnalités de l'application

###  Dashboard Accueil
Statistiques clés du marché, top 5 arrondissements, tendances et indicateurs (prix médian, surface moyenne, rendement locatif).

###  Analyse du Marché
4 onglets interactifs : distribution des prix, analyse par type de bien, séries temporelles et comparatifs par arrondissement.

###  Prédiction de Prix
Formulaire personnalisé (surface, pièces, arrondissement, étage, équipements) avec estimation instantanée, cluster assigné et conseil d'investissement.

###  Carte Interactive
Visualisation Folium avec marqueurs colorés par cluster, popups détaillés et filtres dynamiques.

###  Analyse des Clusters
Radar charts comparatifs, répartition en pie chart, box plots par métrique et export CSV.

###  Modèles IA
Détails K-Means (inertie, itérations), projection PCA 2D, importance des features Random Forest et métriques de performance.

---

##  Architecture

```
projet_immobilier_app/
├── app.py                      # Application Streamlit (point d'entrée)
├── data/
│   ├── raw/                    # Données brutes DVF
│   ├── processed/              # Données nettoyées
│   └── real/paris_final.csv    # Dataset final (2000 biens)
├── models/
│   ├── kmeans_model.pkl        # K-Means (5 clusters)
│   └── price_predictor.pkl     # Random Forest (100 arbres)
├── scripts/
│   ├── download_data.py        # Génération des données
│   ├── train_models.py         # Entraînement des modèles
│   └── create_final_dataset.py # Création dataset final
├── notebooks/
│   ├── 01_EDA.ipynb            # Analyse exploratoire
│   ├── 02_Kmeans.ipynb         # Segmentation
│   └── 03_Prediction.ipynb     # Modélisation
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
# Cloner le repo
git clone https://github.com/Merieem2222/real-estate-analysis.git
cd real-estate-analysis

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Dépendances
pip install -r requirements.txt

# Entraîner les modèles
python scripts/train_models.py

# Lancer l'application
streamlit run app.py
```

---

##  Importance des features (Random Forest)

```
Surface ████████████████████████████ 28%
Arrondissement ██████████████████████ 22%
Prix/m² ██████████████████ 18%
Pièces ████████████ 12%
Année construction ████████ 8%
Ascenseur █████ 5%
Balcon ███ 3%
Terrasse ██ 2%
Étage ██ 2%
```

---

##  Stack technique

| Composant | Technologies |
|-----------|-------------|
| **ML** | Scikit-learn (K-Means, Random Forest, PCA) |
| **Data** | Pandas, NumPy |
| **Dashboard** | Streamlit |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Cartographie** | Folium |
| **Serialisation** | Pickle |
| **Source données** | DVF 2024 (calibration) |

---

##  Ce que j'ai appris

- Construire un **pipeline data science complet** : collecte, nettoyage, modélisation, déploiement
- Appliquer le **clustering K-Means** pour la segmentation de marché
- Obtenir un **R² de 0.989** avec Random Forest et feature engineering adapté
- Créer un **dashboard interactif professionnel** avec Streamlit et Plotly
- Intégrer des **cartes géographiques** Folium dans une application web
- Interpréter les résultats ML en **recommandations business**

---

##  Auteure

**Meriem DABDOUBI** — Étudiante B3 Data & IA @ ECE Paris  
[GitHub](https://github.com/Merieem2222)

---

*Projet réalisé dans le cadre de ma formation en Data & Intelligence Artificielle à l'ECE Paris — 2025*
