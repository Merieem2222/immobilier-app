# src/clustering.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

class MarketSegmenter:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.inertia = []
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """Détermine le nombre optimal de clusters"""
        self.inertia = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            self.inertia.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) > 1:
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Méthode du coude
        self.plot_elbow_curve(max_clusters)
        
        # Meilleur score de silhouette
        optimal_k = np.argmax(silhouette_scores) + 2
        
        return optimal_k
    
    def plot_elbow_curve(self, max_clusters):
        """Affiche la courbe du coude"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), self.inertia, 'bo-')
        plt.xlabel('Nombre de clusters')
        plt.ylabel('Inertie')
        plt.title('Méthode du coude pour K-means')
        plt.grid(True)
        plt.show()
    
    def segment_market(self, X):
        """Segmente le marché avec K-means"""
        if self.n_clusters is None:
            self.n_clusters = self.find_optimal_clusters(X)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=42, 
                            n_init=10)
        labels = self.kmeans.fit_predict(X)
        
        return labels, self.kmeans.cluster_centers_
    
    def analyze_clusters(self, df, labels):
        """Analyse les caractéristiques de chaque cluster"""
        df['cluster'] = labels
        
        cluster_analysis = {}
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            analysis = {
                'taille': len(cluster_data),
                'prix_moyen': cluster_data['prix'].mean(),
                'surface_moyenne': cluster_data['surface'].mean(),
                'prix_m2_moyen': cluster_data['prix_m2'].mean(),
                'arrondissement_modal': cluster_data['arrondissement'].mode()[0],
                'caracteristiques': cluster_data.describe().to_dict()
            }
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis