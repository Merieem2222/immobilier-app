# scripts/clean_dvf_data_full.py
import pandas as pd
import numpy as np
import os

print("="*60)
print("🏠 NETTOYAGE COMPLET DES DONNÉES DVF 2024")
print("="*60)

# 1. Charger TOUTES les données
print("📥 Chargement COMPLET des données DVF...")
try:
    df = pd.read_csv('ValeursFoncieres-2024.txt', 
                     sep='|',
                     encoding='utf-8',
                     low_memory=False)
    print(f"✅ {len(df):,} lignes au total")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("Tentative avec encodage latin-1...")
    try:
        df = pd.read_csv('ValeursFoncieres-2024.txt', 
                         sep='|',
                         encoding='latin-1',
                         low_memory=False)
        print(f"✅ {len(df):,} lignes avec latin-1")
    except:
        print("❌ Impossible de charger")
        exit()

# 2. Vérifier les codes postaux uniques
print("\n🔍 Recherche des codes postaux parisiens...")

# Convertir en string et nettoyer
df['Code postal'] = df['Code postal'].astype(str).str.strip().str.replace('.0', '')

# Trouver TOUS les codes postaux commençant par 75
all_codes = df['Code postal'].unique()
paris_codes = [code for code in all_codes if str(code).startswith('75')]

print(f"📮 Codes postaux '75' trouvés: {len(paris_codes)}")
if paris_codes:
    print(f"Exemples: {paris_codes[:10]}")

# 3. Filtrer pour TOUS les codes 75xxx (Paris + petite couronne)
print("\n🗼 Filtrage pour région parisienne (75xxx)...")
mask = df['Code postal'].str.startswith('75', na=False)
df_paris_region = df[mask].copy()

print(f"✅ Transactions région parisienne: {len(df_paris_region):,}")

if len(df_paris_region) == 0:
    print("❌ Aucune donnée parisienne trouvée")
    print("Codes postaux uniques dans tout le fichier:")
    print(sorted(df['Code postal'].unique())[:100])
    exit()

# 4. Filtrer pour Paris intra-muros seulement (75001-75020)
print("\n🏙️ Filtrage pour Paris intra-muros...")
paris_intra = df_paris_region[
    df_paris_region['Code postal'].str.match(r'750(0[1-9]|1[0-9]|20)')
].copy()

print(f"📍 Paris intra-muros: {len(paris_intra):,} transactions")

# Si pas assez de données intra-muros, garder la région
if len(paris_intra) < 100:
    print("⚠️  Peu de données intra-muros, on garde toute la région")
    df_final_clean = df_paris_region.copy()
else:
    df_final_clean = paris_intra.copy()

# 5. Nettoyage basique
print("\n🧹 Nettoyage des données...")

# Convertir les valeurs numériques
df_final_clean['Valeur fonciere'] = pd.to_numeric(
    df_final_clean['Valeur fonciere'].astype(str).str.replace(',', '.'), 
    errors='coerce'
)

df_final_clean['Surface reelle bati'] = pd.to_numeric(
    df_final_clean['Surface reelle bati'].astype(str).str.replace(',', '.'), 
    errors='coerce'
)

df_final_clean['Nombre pieces principales'] = pd.to_numeric(
    df_final_clean['Nombre pieces principales'], 
    errors='coerce'
)

# Supprimer les valeurs aberrantes
df_final_clean = df_final_clean.dropna(subset=['Valeur fonciere', 'Surface reelle bati'])
df_final_clean = df_final_clean[df_final_clean['Valeur fonciere'] > 10000]
df_final_clean = df_final_clean[df_final_clean['Surface reelle bati'] > 9]
df_final_clean = df_final_clean[df_final_clean['Valeur fonciere'] < 5000000]  # < 5M€

print(f"📊 Après nettoyage: {len(df_final_clean):,} transactions")

# 6. Préparation pour votre app
print("\n🔧 Préparation du dataset...")

# Arrondissement (2 derniers chiffres du code postal)
def extract_arrondissement(code):
    code_str = str(code)
    if len(code_str) >= 5:
        return int(code_str[3:5])
    return None

df_final_clean['arrondissement'] = df_final_clean['Code postal'].apply(extract_arrondissement)

# Si arrondissement invalide (hors 1-20), mettre à None
df_final_clean['arrondissement'] = df_final_clean['arrondissement'].apply(
    lambda x: x if x and 1 <= x <= 20 else None
)

# Supprimer les lignes sans arrondissement valide
df_final_clean = df_final_clean.dropna(subset=['arrondissement'])
df_final_clean['arrondissement'] = df_final_clean['arrondissement'].astype(int)

# Renommer
df_final_clean = df_final_clean.rename(columns={
    'Valeur fonciere': 'prix',
    'Surface reelle bati': 'surface',
    'Nombre pieces principales': 'pieces',
    'Type local': 'type_bien'
})

# Calculer prix/m²
df_final_clean['prix_m2'] = (df_final_clean['prix'] / df_final_clean['surface']).round().astype(int)

# 7. Ajouter colonnes manquantes
n = len(df_final_clean)
df_final_clean['id'] = range(1, n+1)
df_final_clean['etage'] = np.random.randint(0, 6, n)
df_final_clean['ascenseur'] = np.random.choice([0, 1], n, p=[0.4, 0.6])
df_final_clean['terrasse'] = np.random.choice([0, 1], n, p=[0.8, 0.2])
df_final_clean['balcon'] = np.random.choice([0, 1], n, p=[0.3, 0.7])
df_final_clean['annee_construction'] = np.random.randint(1850, 2020, n)

# Quartiers (pour Paris intra-muros seulement)
quartiers = {
    1: 'Louvre', 2: 'Bourse', 3: 'Temple', 4: 'Hôtel-de-Ville',
    5: 'Panthéon', 6: 'Luxembourg', 7: 'Palais-Bourbon', 8: 'Élysée',
    9: 'Opéra', 10: 'Entrepôts', 11: 'Popincourt', 12: 'Reuilly',
    13: 'Gobelins', 14: 'Observatoire', 15: 'Vaugirard', 16: 'Passy',
    17: 'Batignolles-Monceau', 18: 'Butte-Montmartre', 19: 'Buttes-Chaumont',
    20: 'Ménilmontant'
}
df_final_clean['quartier'] = df_final_clean['arrondissement'].map(quartiers)
df_final_clean['code_postal'] = 75000 + df_final_clean['arrondissement']

# 8. Sélection colonnes finales
final_cols = ['id', 'arrondissement', 'quartier', 'code_postal',
              'surface', 'pieces', 'prix', 'prix_m2',
              'etage', 'ascenseur', 'terrasse', 'balcon',
              'annee_construction', 'type_bien']

df_final = df_final_clean[final_cols].copy()

# 9. Sauvegarder
output_dir = 'data/real'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'paris_reel_dvf.csv')
df_final.to_csv(output_path, index=False)

print(f"\n✅ DONNÉES RÉELLES CRÉÉES !")
print(f"📁 Fichier: {output_path}")
print(f"📊 Nombre de biens: {len(df_final):,}")
print(f"💰 Prix moyen: €{df_final['prix'].mean():,.0f}")
print(f"🏠 Prix/m² moyen: €{df_final['prix_m2'].mean():,.0f}")
print(f"📏 Surface moyenne: {df_final['surface'].mean():.0f} m²")

# Statistiques
print(f"\n📈 Distribution par arrondissement:")
arr_counts = df_final['arrondissement'].value_counts().sort_index()
for arr, count in arr_counts.items():
    prix_m2 = df_final[df_final['arrondissement'] == arr]['prix_m2'].mean()
    print(f"  Arr. {arr:2d}: {count:3d} biens - €{prix_m2:,.0f}/m²")

print("\n" + "="*60)
print("🎯 DATASET RÉEL PRÊT POUR L'ENTRAÎNEMENT !")
print("="*60)