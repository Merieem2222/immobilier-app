# scripts/clean_real_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_dvf_data(file_path='ValeursFoncieres-2024.txt'):
    """Charge les données DVF réelles"""
    print(f"📥 Chargement du fichier: {file_path}")
    
    # Lire avec le bon encodage et séparateur
    try:
        df = pd.read_csv(file_path, sep='|', encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep='|', encoding='latin-1', low_memory=False)
        except:
            df = pd.read_csv(file_path, sep='|', encoding='ISO-8859-1', low_memory=False)
    
    print(f"✅ {len(df):,} lignes chargées")
    return df

def filter_paris_transactions(df):
    """Filtre pour garder seulement Paris"""
    print("🔍 Filtrage pour Paris...")
    
    # Créer une colonne code_postal numérique
    df['Code postal'] = pd.to_numeric(df['Code postal'], errors='coerce')
    
    # Garder seulement Paris (75001 à 75020)
    mask = df['Code postal'].between(75001, 75020)
    df_paris = df[mask].copy()
    
    print(f"   → {len(df_paris):,} transactions parisiennes")
    return df_paris

def clean_and_transform(df):
    """Nettoie et transforme les données"""
    print("🧹 Nettoyage des données...")
    
    # 1. Sélectionner et renommer les colonnes importantes
    df_clean = df[[
        'Date mutation', 'Valeur fonciere', 'Code postal',
        'Type local', 'Surface reelle bati', 'Nombre pieces principales',
        'Commune', 'Nature mutation'
    ]].copy()
    
    df_clean.columns = [
        'date_transaction', 'prix', 'code_postal',
        'type_bien', 'surface', 'pieces',
        'commune', 'type_transaction'
    ]
    
    # 2. Extraire l'arrondissement
    df_clean['arrondissement'] = (df_clean['code_postal'] - 75000).astype(int)
    
    # 3. Filtrer les ventes (pas les échanges, partages...)
    df_clean = df_clean[df_clean['type_transaction'] == 'Vente']
    
    # 4. Filtrer les types de biens
    df_clean = df_clean[df_clean['type_bien'].isin(['Appartement', 'Maison'])]
    
    # 5. Convertir les types de données
    df_clean['prix'] = pd.to_numeric(df_clean['prix'], errors='coerce')
    df_clean['surface'] = pd.to_numeric(df_clean['surface'], errors='coerce')
    df_clean['pieces'] = pd.to_numeric(df_clean['pieces'], errors='coerce')
    
    # 6. Filtrer les valeurs aberrantes
    df_clean = df_clean[
        (df_clean['prix'] >= 10000) &  # Prix minimum
        (df_clean['prix'] <= 20000000) &  # Prix maximum
        (df_clean['surface'] >= 10) &  # Surface minimum
        (df_clean['surface'] <= 500) &  # Surface maximum
        (df_clean['pieces'] >= 1) &  # Au moins 1 pièce
        (df_clean['pieces'] <= 10)  # Maximum 10 pièces
    ]
    
    # 7. Calculer le prix au m²
    df_clean['prix_m2'] = (df_clean['prix'] / df_clean['surface']).round().astype(int)
    
    # 8. Filtrer les prix/m² extrêmes
    df_clean = df_clean[
        (df_clean['prix_m2'] >= 3000) &  # Minimum 3k€/m²
        (df_clean['prix_m2'] <= 30000)  # Maximum 30k€/m²
    ]
    
    # 9. Supprimer les valeurs manquantes
    df_clean = df_clean.dropna(subset=['prix', 'surface', 'arrondissement', 'prix_m2'])
    
    # 10. Ajouter les colonnes nécessaires pour votre app
    # (Certaines colonnes ne sont pas dans DVF, on les simule)
    n = len(df_clean)
    
    # Caractéristiques architecturales (simulées mais réalistes)
    df_clean['etage'] = np.random.randint(0, 8, n)
    df_clean['ascenseur'] = np.where(
        (df_clean['etage'] > 2) | (df_clean['prix_m2'] > 10000),
        1,  # Ascenseur probable pour étage > 2 ou prix élevé
        np.random.choice([0, 1], n, p=[0.6, 0.4])
    )
    df_clean['terrasse'] = np.where(
        df_clean['etage'] > 0,
        np.random.choice([0, 1], n, p=[0.4, 0.6]),  # Plus probable avec étage
        np.random.choice([0, 1], n, p=[0.8, 0.2])
    )
    df_clean['balcon'] = np.random.choice([0, 1], n, p=[0.3, 0.7])
    
    # Année de construction (simulée mais cohérente avec l'arrondissement)
    # Les arrondissements centraux ont des bâtiments plus anciens
    df_clean['annee_construction'] = df_clean['arrondissement'].apply(
        lambda x: np.random.randint(1850, 1930) if x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16]
        else np.random.randint(1950, 2020)
    )
    
    # Quartier (simplifié)
    quartiers_par_arr = {
        1: 'Louvre', 2: 'Bourse', 3: 'Temple', 4: 'Hôtel-de-Ville',
        5: 'Panthéon', 6: 'Luxembourg', 7: 'Palais-Bourbon', 8: 'Élysée',
        9: 'Opéra', 10: 'Entrepôts', 11: 'Popincourt', 12: 'Reuilly',
        13: 'Gobelins', 14: 'Observatoire', 15: 'Vaugirard', 16: 'Passy',
        17: 'Batignolles-Monceau', 18: 'Butte-Montmartre', 19: 'Buttes-Chaumont',
        20: 'Ménilmontant'
    }
    df_clean['quartier'] = df_clean['arrondissement'].map(quartiers_par_arr)
    
    # 11. Renommer pour correspondre à votre app
    df_clean = df_clean.rename(columns={
        'type_bien': 'type_bien',
        'pieces': 'pieces'
    })
    
    # 12. Ajouter ID unique
    df_clean['id'] = range(1, len(df_clean) + 1)
    
    # 13. Réorganiser les colonnes
    final_columns = [
        'id', 'arrondissement', 'quartier', 'type_bien', 'surface', 'pieces',
        'prix', 'prix_m2', 'etage', 'ascenseur', 'terrasse', 'balcon',
        'annee_construction', 'date_transaction', 'code_postal', 'commune'
    ]
    
    return df_clean[final_columns]

def save_and_analyze(df, output_dir='data/real'):
    """Sauvegarde et analyse les résultats"""
    print("💾 Sauvegarde des données nettoyées...")
    
    # Créer le dossier
    os.makedirs(output_dir, exist_ok=True)
    
    # Chemin de sortie
    output_path = os.path.join(output_dir, 'immobilier_paris_reel.csv')
    
    # Sauvegarder
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Données sauvegardées dans: {output_path}")
    print(f"📊 {len(df)} transactions valides")
    
    # Statistiques
    print("\n" + "="*60)
    print("📈 STATISTIQUES DES DONNÉES RÉELLES NETTOYÉES")
    print("="*60)
    
    stats = {
        'Prix moyen': f"€{df['prix'].mean():,.0f}",
        'Prix/m² moyen': f"€{df['prix_m2'].mean():,.0f}",
        'Surface moyenne': f"{df['surface'].mean():.1f} m²",
        'Pièces moyennes': f"{df['pieces'].mean():.1f}",
        'Étage moyen': f"{df['etage'].mean():.1f}",
        'Avec ascenseur': f"{df['ascenseur'].mean()*100:.1f}%",
        'Année construction moyenne': f"{df['annee_construction'].mean():.0f}"
    }
    
    for key, value in stats.items():
        print(f"  {key:25} : {value}")
    
    # Par arrondissement
    print("\n🏙️  Prix/m² moyen par arrondissement:")
    print("-" * 50)
    for arr in sorted(df['arrondissement'].unique()):
        subset = df[df['arrondissement'] == arr]
        if len(subset) > 10:  # Afficher seulement si assez de données
            print(f"  Arr. {arr:2d} : {len(subset):4d} biens "
                  f"→ €{subset['prix_m2'].mean():,.0f}/m² "
                  f"(€{subset['prix'].mean():,.0f} total)")
    
    return output_path

def main():
    """Fonction principale"""
    print("="*60)
    print("🔄 NETTOYAGE DES DONNÉES DVF RÉELLES")
    print("="*60)
    
    # 1. Charger les données brutes
    df_raw = load_dvf_data('ValeursFoncieres-2024.txt')
    
    # 2. Filtrer pour Paris
    df_paris = filter_paris_transactions(df_raw)
    
    if len(df_paris) == 0:
        print("❌ Aucune donnée Parisienne trouvée!")
        return None
    
    # 3. Nettoyer et transformer
    df_clean = clean_and_transform(df_paris)
    
    if len(df_clean) == 0:
        print("❌ Aucune donnée valide après nettoyage!")
        return None
    
    # 4. Sauvegarder et analyser
    output_path = save_and_analyze(df_clean)
    
    print("\n" + "="*60)
    print("🎉 NETTOYAGE TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    
    return output_path

