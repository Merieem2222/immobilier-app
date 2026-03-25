#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

def train_models():
    """Entraîne les modèles K-means et Random Forest"""
    print("Chargement des données...")
    
    # Charger les données
    data = pd.read_csv('data/real/paris_reel_dvf.csv')
    
    print(f"{len(data)} biens chargés")
    
    # 1. Entraîner le modèle K-means
    print("\nEntrainement du modèle K-means...")
    cluster_features = ['surface', 'pieces', 'arrondissement', 'prix_m2']
    X_cluster = data[cluster_features]
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_cluster)
    
    print(f"K-means entraîné avec {kmeans.n_clusters} clusters")
    print(f"   Inertie: {kmeans.inertia_:,.2f}")
    
    # 2. Entraîner le modèle de prédiction
    print("\nEntrainement du modèle Random Forest...")
    features = ['surface', 'pieces', 'arrondissement', 'etage', 
               'ascenseur', 'terrasse', 'balcon', 'annee_construction', 'cluster']
    
    X = data[features]
    y = data['prix']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraînement
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Évaluation
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest entraîné avec {rf.n_estimators} arbres")
    print(f"   MAE sur le test set: €{mae:,.0f}")
    print(f"   R² score: {r2:.3f}")
    
    # 3. Sauvegarder les modèles
    print("\nSauvegarde des modèles...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open('models/price_predictor.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    # Sauvegarder les données avec les clusters
    data.to_csv('data/processed/immobilier_paris_with_clusters.csv', index=False)
    
    print("Modèles sauvegardés dans le dossier models/")
    print("\nResume des clusters:")
    for cluster_id in range(5):
        cluster_data = data[data['cluster'] == cluster_id]
        print(f"   Cluster {cluster_id}: {len(cluster_data)} biens, "
              f"prix moyen: €{cluster_data['prix'].mean():,.0f}, "
              f"prix/m²: €{cluster_data['prix_m2'].mean():,.0f}")

if __name__ == "__main__":
    train_models()