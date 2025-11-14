
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Exploration
def load_and_explore_data(file_path):
    """Load the dataset and perform initial exploration."""
    # Load the data
    df = pd.read_csv(file_path)
    
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    print("\nUnique values in categorical columns:")
    print("Month:", df['Month'].unique())
    print("VisitorType:", df['VisitorType'].unique())
    print("Weekend:", df['Weekend'].unique())
    print("Revenue:", df['Revenue'].unique())
    
    print("\nCount of Revenue classes:")
    print(df['Revenue'].value_counts())
    
    return df

# 2. Feature Engineering
def preprocess_data(df):
    """Preprocess the dataset, applying necessary transformations."""
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Boolean columns: Convert to numeric (0/1)
    processed_df['Weekend'] = processed_df['Weekend'].astype(int)
    processed_df['Revenue'] = processed_df['Revenue'].astype(int)
    
    # Mean encoding for 'Month'
    month_mapping = processed_df.groupby('Month')['Revenue'].mean().to_dict()
    processed_df['MonthEncoded'] = processed_df['Month'].map(month_mapping)
    
    # Mean encoding for 'VisitorType'
    visitor_mapping = processed_df.groupby('VisitorType')['Revenue'].mean().to_dict()
    processed_df['VisitorTypeEncoded'] = processed_df['VisitorType'].map(visitor_mapping)
    
    # Drop original categorical columns
    processed_df = processed_df.drop(['Month', 'VisitorType'], axis=1)
    
    return processed_df

# 3. Clustering Functions
def apply_kmeans(X, k=4):
    """Apply K-means clustering with the specified number of clusters."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Get cluster centers for visualization
    centers = kmeans.cluster_centers_
    
    return cluster_labels, centers, kmeans

def apply_agglomerative(X, k=4):
    """Apply Complete-Linkage Agglomerative Hierarchical Clustering."""
    agglom = AgglomerativeClustering(n_clusters=k, linkage='complete')
    cluster_labels = agglom.fit_predict(X)
    
    return cluster_labels, agglom

# 4. Evaluation with Rand Index
def calculate_rand_index(true_labels, cluster_labels):
    """
    Calculate the Rand Index between true labels and cluster labels.
    
    Parameters:
    true_labels: array of binary labels (0/1)
    cluster_labels: array of cluster assignments (0-3 for k=4)
    
    Returns:
    float: Rand Index score
    """
    n = len(true_labels)
    
    # Create all possible pairs of indices
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Sets S and D as defined in the assignment
    S = set()
    D = set()
    
    for i, j in pairs:
        # Check if the pair belongs to set S
        if true_labels[i] == true_labels[j] and cluster_labels[i] == cluster_labels[j]:
            S.add((i, j))
        # Check if the pair belongs to set D
        elif true_labels[i] != true_labels[j] and cluster_labels[i] != cluster_labels[j]:
            D.add((i, j))
    
    # Calculate Rand Index: RI = 2(|S|+|D|) / (n*(n-1))
    rand_index = 2 * (len(S) + len(D)) / (n * (n - 1))
    
    return rand_index

# 5. Analysis Functions
def analyze_clusters(X, labels):
    """
    Analyze cluster characteristics without visualization.
    
    Parameters:
    X: Feature matrix
    labels: Cluster labels
    
    Returns:
    dict: Statistics about the clusters
    """
    # Convert to DataFrame for easier analysis
    df_analysis = pd.DataFrame(X)
    df_analysis['cluster'] = labels
    
    # Calculate basic statistics
    stats = {}
    
    # Size of each cluster
    stats['cluster_sizes'] = df_analysis['cluster'].value_counts().sort_index().to_dict()
    
    # Mean of each feature in each cluster
    stats['cluster_means'] = df_analysis.groupby('cluster').mean().to_dict()
    
    # Standard deviation of each feature in each cluster
    stats['cluster_stds'] = df_analysis.groupby('cluster').std().to_dict()
    
    return stats

def calculate_feature_importance(df, labels):
    """
    Calculate feature importance for each cluster by comparing means.
    
    Parameters:
    df: Original dataframe with features
    labels: Cluster labels
    
    Returns:
    DataFrame: Cluster means for each feature
    """
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Calculate mean values for each feature in each cluster
    cluster_means = df_with_clusters.groupby('Cluster').mean()
    
    # Calculate normalized means to highlight differences
    overall_mean = df.mean()
    normalized_means = cluster_means / overall_mean
    
    return cluster_means, normalized_means

# 6. Main Function
def main():
    # File path
    file_path = 'online_shoppers_intention.csv'
    
    # 1. Load and explore data
    print("Loading and exploring data...")
    df = load_and_explore_data(file_path)
    
    # 2. Preprocess data
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df)
    
    # Separate features and target
    X = processed_df.drop('Revenue', axis=1)
    y = processed_df['Revenue'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Apply clustering algorithms
    print("\nApplying K-means clustering...")
    kmeans_labels, kmeans_centers, kmeans_model = apply_kmeans(X_scaled, k=4)
    
    print("\nApplying Agglomerative clustering...")
    agglom_labels, agglom_model = apply_agglomerative(X_scaled, k=4)
    
    # 4. Calculate Rand Index
    print("\nCalculating Rand Index...")
    kmeans_ri = calculate_rand_index(y, kmeans_labels)
    agglom_ri = calculate_rand_index(y, agglom_labels)
    
    print(f"K-means Rand Index: {kmeans_ri:.4f}")
    print(f"Agglomerative Clustering Rand Index: {agglom_ri:.4f}")
    
    if kmeans_ri > agglom_ri:
        print("K-means clustering performs better based on the Rand Index.")
    else:
        print("Agglomerative clustering performs better based on the Rand Index.")
    
    # 5. Analyze results
    print("\nAnalyzing clusters...")
    # Analyze K-means clusters
    kmeans_stats = analyze_clusters(X_scaled, kmeans_labels)
    print("\nK-means cluster sizes:")
    print(kmeans_stats['cluster_sizes'])
    
    # Analyze Agglomerative clusters
    agglom_stats = analyze_clusters(X_scaled, agglom_labels)
    print("\nAgglomerative cluster sizes:")
    print(agglom_stats['cluster_sizes'])
    
    # Feature importance for K-means
    kmeans_feature_means, kmeans_normalized_means = calculate_feature_importance(X, kmeans_labels)
    print("\nK-means cluster means:")
    print(kmeans_feature_means)
    
    # Feature importance for Agglomerative
    agglom_feature_means, agglom_normalized_means = calculate_feature_importance(X, agglom_labels)
    print("\nAgglomerative cluster means:")
    print(agglom_feature_means)
    
    # 6. Save results to file
    with open("clustering_report.txt", "w") as f:
        f.write("CAP 6610 - Machine Learning (Spring 2025)\n")
        f.write("Assignment 2 - Online Shoppers Intention Clustering\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("=================\n\n")
        
        f.write("1. K-means Clustering\n")
        f.write("-----------------\n")
        f.write(f"Rand Index: {kmeans_ri:.4f}\n\n")
        f.write("Cluster Distribution:\n")
        f.write(str(pd.Series(kmeans_labels).value_counts().sort_index()) + "\n\n")
        
        f.write("2. Agglomerative Clustering\n")
        f.write("-----------------------\n")
        f.write(f"Rand Index: {agglom_ri:.4f}\n\n")
        f.write("Cluster Distribution:\n")
        f.write(str(pd.Series(agglom_labels).value_counts().sort_index()) + "\n\n")
        
        f.write("COMPARISON\n")
        f.write("==========\n\n")
        if kmeans_ri > agglom_ri:
            f.write(f"K-means clustering performs better with a Rand Index of {kmeans_ri:.4f} compared to {agglom_ri:.4f} for Agglomerative clustering.\n")
        else:
            f.write(f"Agglomerative clustering performs better with a Rand Index of {agglom_ri:.4f} compared to {kmeans_ri:.4f} for K-means clustering.\n")
    
    print("\nResults saved to 'clustering_report.txt'")

if __name__ == "__main__":
    main()
