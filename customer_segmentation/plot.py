import matplotlib.pyplot as plt
import seaborn as sns

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