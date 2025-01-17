import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, davies_bouldin_score
from customer_segmentation.plot import visualize_clusters

def scale_minmax(df, column):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])

def evaluate_clusters(df_scaled, df_objective, column):
    print(f"Silhouette = {silhouette_score(df_scaled, df_objective[column]):.4f}\n"
          f"Davies-Bouldin = {davies_bouldin_score(df_scaled, df_objective[column]):.4f}")
    visualize_clusters(df_scaled, df_objective, column)

def filter_by_cluster(df, cluster_value, cluster_column = 'cluster'):
    return df[df[cluster_column] == cluster_value]

def scale_data(df, columns):
    scaler = MinMaxScaler()
    df_scaled = df[columns]
    df_scaled = scaler.fit_transform(df_scaled)
    return df_scaled

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca_result, pca

def pca_loadings(pca_model, columns):
    loadings = pd.DataFrame(pca_model.components_, columns=columns, index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])])
    return loadings

class DBSCANPipeline(Pipeline, BaseEstimator, ClusterMixin):
    def fit(self, X, y=None):
        X = X.select_dtypes(include=[np.number])
        self.steps[0][1].fit(X)
        X_scaled = self.steps[0][1].transform(X)
        self.steps[1][1].fit(X_scaled)
        self.labels_ = self.steps[1][1].labels_
        self.X_scaled = X_scaled
        return self

    def predict(self, X):
        X = X.select_dtypes(include=[np.number])
        X_new_scaled = self.steps[0][1].transform(X)
        datos_validos = self.X_scaled[self.labels_ != -1]
        clusters_validos = self.labels_[self.labels_ != -1]
        cluster_asignaciones = []
        for nuevo in X_new_scaled:
            distancias = np.linalg.norm(datos_validos - nuevo, axis=1)
            vecino_mas_cercano = np.min(distancias)
            if vecino_mas_cercano > self.steps[1][1].eps:
                cluster_asignaciones.append(-1)  # Ruido
            else:
                cluster_asignado = clusters_validos[np.argmin(distancias)]
                cluster_asignaciones.append(int(cluster_asignado))
        X_con_clusters = X.copy()
        X_con_clusters['cluster'] = cluster_asignaciones
        print("\nDatos con clúster asignado:")
        print(X_con_clusters)
        conteo_clusters = X_con_clusters['cluster'].value_counts().sort_index()
        print("\nConteo de registros por clúster:")
        print(conteo_clusters)

        return cluster_asignaciones

def crear_pipeline():
    return DBSCANPipeline([
        ('scaler', MinMaxScaler()),
        ('dbscan', DBSCAN(eps=0.04, min_samples=50))  
    ])