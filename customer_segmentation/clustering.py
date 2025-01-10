import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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
    """Displays the loadings of the PCA components."""
    loadings = pd.DataFrame(pca_model.components_, columns=columns, index=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])])
    return loadings

def dbscan_pipeline(data):

    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    eps = 0.04
    min_samples = 50
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    clusters = dbscan.fit_predict(data_scaled)

    clustered_data = data.copy()
    clustered_data["cluster"] = clusters

    clustered_data = clustered_data[clustered_data["cluster"] != -1]
    valid_clusters = clustered_data["cluster"].values
    data_scaled_filtered = data_scaled.iloc[clustered_data.index]

    num_clusters = len(set(valid_clusters))
    if num_clusters > 1:
        silhouette = silhouette_score(data_scaled_filtered, valid_clusters)
        davies_bouldin = davies_bouldin_score(data_scaled_filtered, valid_clusters)
    else:
        silhouette = -1
        davies_bouldin = -1

    metrics = {
        "Model": "DBSCAN",
        "Silhouette Score": silhouette,
        "Davies-Bouldin Score": davies_bouldin,
        "Num Clusters": num_clusters,
    }

    pipeline = Pipeline([
        ("scaler", scaler),
        ("clustering", dbscan),
    ])

    cluster_summary = (
        clustered_data.groupby("cluster")
        .agg({col: "mean" for col in clustered_data.columns if col != "cluster"})
        .reset_index()
    )
    cluster_counts = clustered_data["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["cluster", "count"]

    cluster_summary = cluster_summary.merge(cluster_counts, on="cluster")

    print("\n=== Métricas del Modelo ===")
    print(metrics)

    print("\n=== Resumen de Clusters ===")
    print(cluster_summary)

    return metrics, pipeline, clustered_data