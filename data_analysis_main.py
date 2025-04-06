import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loader import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Folders maker
os.makedirs("results", exist_ok=True)
os.makedirs("results/boxplot_numeric", exist_ok=True)
os.makedirs("results/violinplot_numeric", exist_ok=True)
os.makedirs("results/histograms_numeric", exist_ok=True)
os.makedirs("results/conditional_histograms_numeric", exist_ok=True)
os.makedirs("results/error_bars_numeric", exist_ok=True)
os.makedirs("results/heatmap_numeric", exist_ok=True)
os.makedirs("results/regression_numeric", exist_ok=True)
os.makedirs("results/dimensionality_reduction", exist_ok=True)


# Converting to categorical values
def convert_to_categorical(df, categorical_columns):
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def convert_to_numerical(df, numerical_columns):
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Nacionality", "International"
]

NUMERIC_COLUMNS = [
     "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Nacionality", "International"
]

# Preparing a script to calculate and save preliminary feature statistics for numeric values
def calculate_numerical_statistics(df):
    numerical_features = df.select_dtypes(include=['number'])
    stats = []

    for column in numerical_features.columns:
        stats.append({
            "Feature": column,
            "Mean": numerical_features[column].mean(),
            "Median": numerical_features[column].median(),
            "Min": numerical_features[column].min(),
            "Max": numerical_features[column].max(),
            "Standard Deviation": numerical_features[column].std(),
            "5th Percentile": numerical_features[column].quantile(0.05),
            "95th Percentile": numerical_features[column].quantile(0.95),
            "Missing Values": numerical_features[column].isnull().sum()
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("results/numerical_statistics.csv", index=False)
    return stats_df


# Preparing a script to calculate and save preliminary feature statistics for categorical values
def calculate_categorical_statistics(df):
    categorical_features = df.select_dtypes(include=['object', 'category'])
    stats = []

    for column in categorical_features.columns:
        class_proportions = categorical_features[column].value_counts(normalize=True).to_dict()
        stats.append({
            "Feature": column,
            "Unique Classes": categorical_features[column].nunique(),
            "Missing Values": categorical_features[column].isnull().sum(),
            "Class Proportions": class_proportions
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("results/categorical_statistics.csv", index=False)
    return stats_df


# Box plots
def generate_boxplots(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        # plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot for {column}')
        # plt.savefig(f'results/boxplot_numeric/{idx}_boxplot.png')
        # plt.close()
        plt.show()


# Violin plots
def generate_violinplots(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        # plt.figure(figsize=(5, 10))
        sns.violinplot(y=df[column])
        plt.title(f'Violinplot for {column}')
        # plt.savefig(f'results/violinplot_numeric/{idx}_violinplot.png')
        # plt.close()
        plt.show()


# Error bars
def generate_error_bars(df):
    np.random.seed(sum(map(ord, "error_bars")))

    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        x = np.random.normal(0, 1, 100)
        f, axs = plt.subplots(2, figsize=(7, 4), sharex=True, constrained_layout=True)

        # Plot pointplot with error bars
        sns.pointplot(x=x, errorbar="sd", capsize=0.3, ax=axs[0])
        axs[0].set_title(f'Point Plot with Error Bars for {column}')

        # Plot stripplot for additional detail
        sns.stripplot(x=x, jitter=0.3, ax=axs[1])
        axs[1].set_title(f'Strip Plot for {column}')

        plt.xlabel(column)
        plt.show()


# Histograms
def generate_histograms(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        # plt.figure(figsize=(10, 10))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram for {column}')
        # plt.savefig(f'results/histograms_numeric/{idx}_histogram.png')
        # plt.close()
        plt.show()


# Conditional histograms
def generate_conditional_histograms(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        # plt.figure(figsize=(10, 10))
        sns.histplot(data=df, x=column, hue="Target", kde=True)
        plt.title(f'Histogram for {column}')
        # plt.savefig(f'results/histograms_numeric/{idx}_histogram.png')
        # plt.close()
        plt.show()


# Heatmaps
def generate_correlation_heatmap(df):
    numerical_features = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 10))
    correlation_matrix = numerical_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap of Feature Correlations')
    # plt.savefig('results/heatmap_numeric/correlation_heatmap.png')
    # plt.close()
    plt.show()


# Regression plots
def generate_regression_plots(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column1 in enumerate(numerical_features.columns):
        g = sns.lmplot(data=df, x=column1, y='Target')
        g.fig.suptitle(f'Linear Regression: {column1} vs Target')
        # g.savefig(f'results/regression_numeric/{idx1}_{idx2}_regression.png')
        # plt.close('all')
        plt.show()


# PSA dimensionality reduction
def perform_pca(df):
    numerical_features = df.select_dtypes(include=['number'])
    if len(numerical_features.columns) > 1:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numerical_features)
        # plt.figure(figsize=(10, 5))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
        plt.title('PCA Visualization')
        # plt.savefig('results/dimensionality_reduction/pca_visualization.png')
        # plt.close()
        plt.show()


# t-SNE dimensionality reduction
def perform_tsne(df, target_column='Target'):
    numerical_features = df.select_dtypes(include=['number'])

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")

    if len(numerical_features.columns) > 1:
        tsne = TSNE(n_components=2, perplexity=50, random_state=42)
        tsne_result = tsne.fit_transform(numerical_features)

        tsne_df = df.copy()
        tsne_df['TSNE-1'] = tsne_result[:, 0]
        tsne_df['TSNE-2'] = tsne_result[:, 1]

        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=tsne_df, x='TSNE-1', y='TSNE-2', hue=target_column, palette='viridis')
        plt.title('t-SNE Visualization')
        plt.show()


if __name__ == '__main__':
    input_file = sys.argv[1]

    try:

        data = load_data(input_file)

        df = convert_to_categorical(data, CATEGORICAL_COLUMNS)
        # encoder = LabelEncoder()
        # data['Target'] = encoder.fit_transform(data['Target'])

        calculate_numerical_statistics(data)
        calculate_categorical_statistics(data)
        # generate_boxplots(data)
        # generate_violinplots(data)
        # generate_error_bars(data)
        # generate_histograms(data)
        # generate_conditional_histograms(data)
        # generate_correlation_heatmap(data)
        # generate_regression_plots(data)
        # perform_pca(data)
        # perform_tsne(data)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
