# src/modeling.py
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

class PricePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'xgboost': xgb.XGBRegressor(random_state=42)
        }
        self.best_model = None
    
    def train_models(self, X, y, test_size=0.2):
        """Entraîne plusieurs modèles et sélectionne le meilleur"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        
        for name, model in self.models.items():
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédiction
            y_pred = model.predict(X_test)
            
            # Évaluation
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
        
        # Sélection du meilleur modèle basé sur le RMSE
        best_model_name = min(results, key=lambda x: results[x]['rmse'])
        self.best_model = results[best_model_name]['model']
        
        return results, X_test, y_test
    
    def predict_with_clusters(self, X, cluster_labels):
        """Prédit les prix en tenant compte des clusters"""
        # Ajout du cluster comme feature
        X_with_cluster = X.copy()
        X_with_cluster['cluster'] = cluster_labels
        
        return self.train_models(X_with_cluster, y)
    
    def feature_importance(self, model, feature_names):
        """Affiche l'importance des features"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Importance des caractéristiques")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
    
    def create_prediction_pipeline(self, df):
        """Pipeline complet de prédiction"""
        # Préparation des données
        preprocessor = DataPreprocessor()
        cleaned_df = preprocessor.clean_data(df)
        prepared_data = preprocessor.prepare_features(cleaned_df)
        
        # Segmentation
        segmenter = MarketSegmenter(n_clusters=4)
        labels, centers = segmenter.segment_market(prepared_data['clustering'])
        cluster_analysis = segmenter.analyze_clusters(cleaned_df, labels)
        
        # Prédiction
        X = prepared_data['prediction']
        y = prepared_data['target']
        X['cluster'] = labels  # Ajout du cluster comme feature
        
        results, X_test, y_test = self.train_models(X, y)
        
        return {
            'results': results,
            'cluster_analysis': cluster_analysis,
            'labels': labels,
            'test_data': (X_test, y_test),
            'preprocessor': preprocessor
        }