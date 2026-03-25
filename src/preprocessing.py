# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def clean_data(self, df):
        """Nettoie et prépare les données"""
        # Suppression des doublons
        df = df.drop_duplicates()
        
        # Gestion des valeurs aberrantes
        df = self.remove_outliers(df, 'prix')
        df = self.remove_outliers(df, 'surface')
        
        # Imputation des valeurs manquantes
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
        
        return df
    
    def remove_outliers(self, df, column):
        """Supprime les outliers avec la méthode IQR"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def prepare_features(self, df):
        """Prépare les features pour le clustering et la prédiction"""
        # Features pour le clustering
        clustering_features = [
            'surface', 'pieces', 'arrondissement', 
            'prix_m2', 'stations_metro'
        ]
        
        # Features pour la prédiction
        prediction_features = clustering_features + [
            'etage', 'ascenseur', 'terrasse', 'annee_construction'
        ]
        
        # Normalisation pour K-means
        X_cluster = df[clustering_features].copy()
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)
        
        return {
            'clustering': X_cluster_scaled,
            'prediction': df[prediction_features],
            'target': df['prix'],
            'df': df
        }