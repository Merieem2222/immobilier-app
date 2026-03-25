# scripts/clean_real_dvf_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime

print("="*60)
print("🏠 NETTOYAGE DES DONNÉES RÉELLES DVF 2024 - PARIS")
print("="*60)

# 1. Charger vos vraies données DVF
print("📥 Chargement de ValeursFoncieres-2024.txt...")
try:
    # Le séparateur est bien | (pipe)
    df_real = pd.read_csv('ValeursFoncieres-2024.txt', 
                          sep='|',
                          encoding='utf-8',
                          low_memory=False,
                          decimal=',')  # Les nombres ont des virgules comme séparateur décimal
    
    print(f"✅ Données chargées : {len(df_real):,} transactions")
    print(f"📊 Colonnes disponibles : {len(df_real.columns)}")
    print(f"📅 Période : {df_real['Date mutation'].min()} à {df_real['Date mutation'].max()}")
    
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    exit()

# 2. Filtrer pour Paris seulement (codes postaux 75001 à 75020)
print("\n🗼 Filtrage pour Paris...")

# Convertir Code postal en string et nettoyer
df_real['Code postal'] = df_real['Code postal'].astype(str).str.strip()

# Filtrer les transactions parisiennes
df_paris = df_real[df_real['Code postal'].str.startswith('750')].copy()

# Vérifier qu'on a des données
if len(df_paris) == 0:
    print("❌ Aucune transaction parisienne trouvée")
    print("Codes postaux uniques dans les données :")
    print(df_real['Code postal'].unique()[:20])
    exit()

print(f"✅ Transactions parisiennes : {len(df_paris):,}")

# 3. Nettoyage des données
print("\n🧹 Nettoyage et préparation...")

# Extraire l'arrondissement du code postal
df_paris['arrondissement'] = df_paris['Code postal'].str[-2:].astype(int)
print(f"📍 Arrondissements trouvés : {sorted(df_paris['arrondissement'].unique())}")

# Convertir la valeur foncière en numérique (les virgules sont des séparateurs décimaux)
df_paris['Valeur fonciere'] = pd.to_numeric(df_paris['Valeur fonciere'].str.replace(',', '.'), errors='coerce')

# Convertir surface en numérique
if 'Surface reelle bati' in df_paris.columns:
    df_paris['Surface reelle bati'] = pd.to_numeric(df_paris['Surface reelle bati'].str.replace(',', '.'), errors='coerce')

# 4. Filtrer seulement les biens intéressants pour votre app
print("\n🔍 Filtrage des types de biens...")

# Garder seulement certains types de locaux
types_valides = ['Appartement', 'Maison', 'Dépendance']
if 'Type local' in df_paris.columns:
    df_paris = df_paris[df_paris['Type local'].isin(types_valides)].copy()
    print(f"✅ Types de biens gardés : {df_paris['Type local'].unique()}")

# Enlever les transactions trop petites (erreurs de saisie probable)
df_paris = df_paris[df_paris['Valeur fonciere'] > 10000].copy()  # Plus de 10k€
df_paris = df_paris[df_paris['Surface reelle bati'] > 9].copy()  # Plus de 9m²

print(f"📊 Transactions après filtrage : {len(df_paris):,}")

# 5. Renommer et créer les colonnes pour votre application
print("\n🔧 Préparation des features...")

# Renommer les colonnes
df_paris.rename(columns={
    'Valeur fonciere': 'prix',
    'Surface reelle bati': 'surface',
    'Nombre pieces principales': 'pieces',
    'Type local': 'type_bien'
}, inplace=True)

# Créer un ID
df_paris = df_paris.reset_index(drop=True)
df_paris['id'] = df_paris.index + 1

# 6. Calculer le prix au m²
df_paris['prix_m2'] = (df_paris['prix'] / df_paris['surface']).round().astype(int)

# 7. Ajouter des features manquantes (simulées mais basées sur réalité)
print("🎲 Ajout de features manquantes (simulées mais réalistes)...")

# Étage : pour appartements, aléatoire mais corrélé avec l'arrondissement
def generate_etage(arrondissement, type_bien):
    if type_bien == 'Appartement':
        if arrondissement <= 8:  # Centre = plus d'étages
            return np.random.randint(0, 7)
        else:  # Périphérie = moins d'étages
            return np.random.randint(0, 4)
    else:  # Maison/Dépendance
        return 0

df_paris['etage'] = df_paris.apply(lambda x: generate_etage(x['arrondissement'], x['type_bien']), axis=1)

# Ascenseur : plus probable dans immeubles récents et étages élevés
df_paris['ascenseur'] = np.where(
    (df_paris['type_bien'] == 'Appartement') & (df_paris['etage'] > 2),
    np.random.choice([0, 1], len(df_paris), p=[0.3, 0.7]),
    np.random.choice([0, 1], len(df_paris), p=[0.8, 0.2])
)

# Terrasse/Balcon : corrélé avec étage et prix
df_paris['terrasse'] = np.where(
    (df_paris['etage'] > 2) & (df_paris['prix_m2'] > 8000),
    np.random.choice([0, 1], len(df_paris), p=[0.6, 0.4]),
    np.random.choice([0, 1], len(df_paris), p=[0.9, 0.1])
)

df_paris['balcon'] = np.where(
    df_paris['type_bien'] == 'Appartement',
    np.random.choice([0, 1], len(df_paris), p=[0.2, 0.8]),
    np.random.choice([0, 1], len(df_paris), p=[0.9, 0.1])
)

# Année de construction : estimée basée sur l'arrondissement
def estimate_construction_year(arrondissement):
    if arrondissement <= 9:  # Centre = plus vieux
        return np.random.randint(1850, 1930)
    elif arrondissement <= 16:  # Moyen = intermédiaire
        return np.random.randint(1930, 1980)
    else:  # Périphérie = plus récent
        return np.random.randint(1980, 2015)

df_paris['annee_construction'] = df_paris['arrondissement'].apply(estimate_construction_year)

# 8. Ajouter les noms des quartiers
print("🗺️ Ajout des quartiers parisiens...")

quartiers = {
    1: 'Louvre', 2: 'Bourse', 3: 'Temple', 4: 'Hôtel-de-Ville',
    5: 'Panthéon', 6: 'Luxembourg', 7: 'Palais-Bourbon', 8: 'Élysée',
    9: 'Opéra', 10: 'Entrepôts', 11: 'Popincourt', 12: 'Reuilly',
    13: 'Gobelins', 14: 'Observatoire', 15: 'Vaugirard', 16: 'Passy',
    17: 'Batignolles-Monceau', 18: 'Butte-Montmartre', 19: 'Buttes-Chaumont',
    20: 'Ménilmontant'
}
df_paris['quartier'] = df_paris['arrondissement'].map(quartiers)
df_paris['code_postal'] = 75000 + df_paris['arrondissement']

# 9. Sélectionner et ordonner les colonnes finales
print("📋 Finalisation du dataset...")

final_columns = [
    'id', 'arrondissement', 'quartier', 'code_postal',
    'surface', 'pieces', 'prix', 'prix_m2',
    'etage', 'ascenseur', 'terrasse', 'balcon',
    'annee_construction', 'type_bien'
]

# Garder seulement les colonnes qui existent
existing_columns = [col for col in final_columns if col in df_paris.columns]
df_final = df_paris[existing_columns].copy()

# 10. Sauvegarder
output_dir = 'data/real'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'paris_reel_dvf_2024.csv')
df_final.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✅ DONNÉES RÉELLES SAUVEGARDÉES !")
print(f"📁 Fichier: {output_path}")
print(f"📊 Nombre de biens: {len(df_final):,}")
print(f"💰 Prix moyen réel: €{df_final['prix'].mean():,.0f}")
print(f"🏠 Prix/m² moyen réel: €{df_final['prix_m2'].mean():,.0f}")
print(f"📈 Surface moyenne: {df_final['surface'].mean():.0f} m²")
print(f"🏘️ Types de biens: {df_final['type_bien'].value_counts().to_dict()}")

# 11. Statistiques par arrondissement
print("\n📊 STATISTIQUES PAR ARRONDISSEMENT:")
stats_by_arr = df_final.groupby('arrondissement').agg({
    'prix': 'mean',
    'prix_m2': 'mean',
    'surface': 'mean',
    'id': 'count'
}).round(0)

stats_by_arr.columns = ['Prix moyen', 'Prix/m² moyen', 'Surface moyenne', 'Nombre de biens']
print(stats_by_arr)

print("\n" + "="*60)
print("🎯 DONNÉES RÉELLES PRÊTES POUR VOTRE APPLICATION !")
print("="*60)
print("""
Commandes suivantes:
1. python scripts/train_models.py  (entraîne avec les vraies données)
2. streamlit run app.py            (lance l'application)
""")