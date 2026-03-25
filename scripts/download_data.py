#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def generate_sample_data(n_samples=1000):
    """Genere des donnees immobilieres d'exemple pour Paris"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'arrondissement': np.random.randint(1, 21, n_samples),
        'surface': np.random.randint(20, 200, n_samples),
        'pieces': np.random.randint(1, 6, n_samples),
        'chambres': np.random.randint(1, 5, n_samples),
        'etage': np.random.randint(0, 10, n_samples),
        'ascenseur': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'terrasse': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'balcon': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'annee_construction': np.random.randint(1850, 2023, n_samples),
        'type_bien': np.random.choice(['Appartement', 'Maison', 'Studio'], n_samples),
        'quartier': np.random.choice(['Saint-Germain', 'Le Marais', 'Montmartre', 
                                    'Bastille', 'Champs-Elysees', 'Odeon'], n_samples),
        'statut': np.random.choice(['Vente', 'Location'], n_samples, p=[0.7, 0.3]),
        'meuble': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
    
    # CORRECTION : Logique de prix plus réaliste
    # Prix de base par m² selon l'arrondissement
    prix_m2_base = {
        1: 15000, 2: 12000, 3: 11000, 4: 14000, 5: 13000,
        6: 16000, 7: 17000, 8: 18000, 9: 10000, 10: 9000,
        11: 8500, 12: 8000, 13: 7500, 14: 8500, 15: 9000,
        16: 15000, 17: 10000, 18: 7000, 19: 6500, 20: 6000
    }
    
    # Prix de base
    data['prix_m2_base'] = data['arrondissement'].map(prix_m2_base)
    
    # Modifications selon les caractéristiques
    data['prix_m2'] = data['prix_m2_base'] + \
                      data['pieces'] * 500 + \
                      data['ascenseur'] * 1000 + \
                      data['terrasse'] * 1500 + \
                      data['balcon'] * 800 + \
                      ((2023 - data['annee_construction']) / 100) * -200  # Vieillissement
    
    # Ajout de variabilité
    data['prix_m2'] = data['prix_m2'] * np.random.uniform(0.9, 1.1, n_samples)
    
    # Calcul du prix total
    data['prix'] = data['prix_m2'] * data['surface']
    
    # Ajout d'un peu de bruit
    data['prix'] = data['prix'] * np.random.uniform(0.95, 1.05, n_samples)
    
    return data

if __name__ == "__main__":
    print("Regeneration des donnees avec correction...")
    
    # Creer le dossier data s'il n'existe pas
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Generer les donnees corrigees
    data = generate_sample_data(2000)
    
    # Sauvegarder
    data.to_csv('data/raw/immobilier_paris_corrige.csv', index=False)
    data.to_csv('data/processed/immobilier_paris_clean.csv', index=False)
    
    print(f"{len(data)} enregistrements generes")
    print("Statistiques corrigees:")
    print(f"   Prix moyen: €{data['prix'].mean():,.0f}")
    print(f"   Prix/m² moyen: €{data['prix_m2'].mean():,.0f}")
    print(f"   Surface moyenne: {data['surface'].mean():.1f} m²")
    
    # Vérification de cohérence
    print("\nVerification de coherence:")
    for i in range(5):
        sample = data.iloc[i]
        prix_calc = sample['prix_m2'] * sample['surface']
        print(f"   Bien {i}: Prix={sample['prix']:,.0f}€, "
              f"Prix/m²={sample['prix_m2']:,.0f}€, "
              f"Surface={sample['surface']}m², "
              f"Prix calculé={prix_calc:,.0f}€")
    
    print("\nDonnees sauvegardees dans data/raw/ et data/processed/")

    