import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

def plot_boxplot(df):
    df_numerical = df.select_dtypes(include=["number"])
    plt.figure(figsize=(12, 6))
    df_numerical.boxplot()
    plt.title("Boxplot of DataFrame")
    plt.show()

def plot_hist(df, column):
    plt.figure(figsize=(6, 4))
    plt.hist(df[column])
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x_column], df[y_column], alpha=0.5, s=10)
    plt.title(f'Scatter plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def plot_correlation_matrix(df, method):
    df_numerical = df.select_dtypes(include=["number"])
    correlation_matrix = df_numerical.corr(method=method)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"Correlation Matrix ({method.capitalize()} Method)")
    plt.show()

def visualize_dendrogram(df_scaled, method):
    linked_sample = linkage(df_scaled, method)
    plt.figure(figsize=(10, 7))
    dendrogram(linked_sample, truncate_mode="level", p=5)
    plt.title(f'Dendrogram for Sample of {df_scaled.shape[0]} Records')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()

def visualize_clusters(df_scaled, df_objective, column):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    plt.figure(figsize=(6, 4))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=df_objective[column], cmap='viridis', s=1)
    plt.title('Cluster Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_kde_by_cluster(df, column, cluster_column='cluster'):
    plt.figure(figsize=(6, 4))
    for cluster in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster]
        sns.kdeplot(cluster_data[column], label=f'Cluster {cluster}', fill=True, common_norm=False)
    plt.title(f'Density Plot of {column} by {cluster_column}', fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title=cluster_column, title_fontsize='13', fontsize='11')
    plt.show()

def plot_boxplot_by_cluster(df, column, cluster_column='cluster'):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=cluster_column, y=column, data=df, hue=cluster_column, palette='Set2', dodge=False)
    plt.title(f'Boxplot of {column} by {cluster_column}', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend([], [], frameon=False)
    plt.show()

def plot_correlation_heatmap_by_cluster(df, cluster_column='cluster'):
    clusters = df[cluster_column].unique()
    for cluster in clusters:
        cluster_data = df[df[cluster_column] == cluster].drop(columns=["cluster"])
        corr_matrix = cluster_data.corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap for Cluster {cluster}', fontsize=14)
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Variables', fontsize=12)
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10, rotation=45)
        plt.show()

def plot_pca(pca_result, cluster_value):
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_value
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue='Cluster', palette='Set2', legend='full')
    plt.title(f'PCA of Cluster {cluster_value}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

def plot_pairwise_scatter(df, cluster_column, variable_1, variable_2):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=variable_1, y=variable_2, hue=cluster_column, palette='Set2')
    plt.title(f'Scatter Plot of {variable_1} vs {variable_2} by {cluster_column}')
    plt.xlabel(variable_1)
    plt.ylabel(variable_2)
    plt.legend(title=cluster_column, title_fontsize='13', fontsize='11')
    plt.show()