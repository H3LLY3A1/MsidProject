import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
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


CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "International", "Target"
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
    numerical_features = df.select_dtypes(include=['number'])
    for idx, column in enumerate(numerical_features.columns):
        # plt.figure(figsize=(5, 10))
        sns.pointplot(y=df[column])
        plt.title(f'Error Bars for {column}')
        # plt.savefig(f'results/error_bars_numeric/{idx}_error_bars.png')
        # plt.close()
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
    categorical_features = df.select_dtypes(include=['object', 'category'])
    for idx, column in enumerate(numerical_features.columns):
        for idx1, cat_col in enumerate(categorical_features.columns):
            # plt.figure(figsize=(10, 10))
            sns.histplot(data=df, x=column, hue=cat_col, kde=True, element='step')
            plt.title(f'Conditional Histogram for {column} by {cat_col}')
            # plt.savefig(f'results/conditional_histograms_numeric/{idx}_{idx1}_conditional_histogram.png')
            # plt.close()
            plt.show()


# Heatmaps
def generate_correlation_heatmap(df):
    numerical_features = df.select_dtypes(include=['number'])
    plt.figure(figsize=(25, 25))
    correlation_matrix = numerical_features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap of Feature Correlations')
    # plt.savefig('results/heatmap_numeric/correlation_heatmap.png')
    # plt.close()
    plt.show()


# Regression plots
def generate_regression_plots(df):
    numerical_features = df.select_dtypes(include=['number'])
    for idx1, column1 in enumerate(numerical_features.columns):
        for idx2, column2 in enumerate(numerical_features.columns[idx1 + 1:]):
            g = sns.lmplot(data=df, x=column1, y=column2)
            g.fig.suptitle(f'Linear Regression: {column1} vs {column2}')
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
def perform_tsne(df):
    numerical_features = df.select_dtypes(include=['number'])
    if len(numerical_features.columns) > 1:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_result = tsne.fit_transform(numerical_features)
        # plt.figure(figsize=(10, 5))
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1])
        plt.title('t-SNE Visualization')
        # plt.savefig('results/dimensionality_reduction/tsne_visualization.png')
        # plt.close()
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
