# src/data_collection.py
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time

class DataCollector:
    def __init__(self):
        self.data_sources = {
            "seloger": "https://api.seloger.com",
            "pap": "https://www.pap.fr",
            "leboncoin": "https://api.leboncoin.fr",
            "data_gouv": "https://www.data.gouv.fr"
        }
    
    def collect_paris_data(self, pages=10):
        """Collecte des données immobilières pour Paris"""
        # Simulation de collecte (en pratique, utiliser les APIs ou web scraping)
        # Pour ce projet, nous utiliserons un dataset simulé ou public
        
        # Exemple de structure des données
        data = {
            'surface': [45, 60, 75, 30, 90, 110, 55, 70],
            'prix': [350000, 450000, 550000, 250000, 650000, 750000, 400000, 500000],
            'arrondissement': [1, 5, 7, 18, 15, 16, 3, 6],
            'pieces': [2, 3, 3, 1, 4, 4, 2, 3],
            'etage': [2, 1, 3, 0, 2, 1, 4, 2],
            'ascenseur': [1, 0, 1, 0, 1, 1, 0, 1],
            'terrasse': [0, 1, 1, 0, 0, 1, 0, 1],
            'annee_construction': [1990, 1930, 2010, 1975, 2005, 1985, 1960, 1995]
        }
        
        return pd.DataFrame(data)
    
    def add_external_features(self, df):
        """Ajoute des données externes (transports, écoles, etc.)"""
        # Données des stations de métro par arrondissement
        metro_stats = {
            1: 15, 2: 10, 3: 8, 4: 12, 5: 20, 6: 14, 
            7: 18, 8: 16, 9: 12, 10: 22, 11: 19, 12: 15,
            13: 25, 14: 18, 15: 20, 16: 14, 17: 16, 18: 12,
            19: 15, 20: 18
        }
        
        df['stations_metro'] = df['arrondissement'].map(metro_stats)
        df['prix_m2'] = df['prix'] / df['surface']
        
        return df