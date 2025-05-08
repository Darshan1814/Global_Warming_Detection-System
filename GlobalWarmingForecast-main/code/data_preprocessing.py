import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, shapiro, kurtosis, skew

# Load the dataset
df = pd.read_csv("global_warming_sim_dataset.csv")

# Summary statistics for an overview
print("\nDetailed Summary Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values Check:")
print(df.isnull().sum())

# Fill missing values (if any)
if df.isnull().sum().any():
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print("Missing values filled using forward and backward fill methods.")

# Homogeneity check for all numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns

print("\nHomogeneity Check for Numerical Columns (Levene's Test):")
homogeneity_results = {}
for col in numerical_columns:
    stat, p = levene(df[col], df[col].mean())
    homogeneity_results[col] = {"Levene's Stat": stat, "p-value": p, "Homogeneous": p > 0.05}

homogeneity_df = pd.DataFrame(homogeneity_results).T
print(homogeneity_df)

# Advanced statistical analysis
print("\nAdvanced Statistical Metrics:")
for col in numerical_columns:
    col_data = df[col]
    skewness = skew(col_data)
    kurt = kurtosis(col_data)
    normality_stat, normality_p = shapiro(col_data)

    print(f"\nColumn: {col}")
    print(f"  Skewness: {skewness:.2f} (Should be near 0 for symmetry)")
    print(f"  Kurtosis: {kurt:.2f} (Should be near 3 for normal distribution)")
    print(f"  Shapiro-Wilk Test p-value: {normality_p:.5f} (p > 0.05 indicates normality)")

# Visualizing distributions
print("\nPlotting Distributions for Numerical Columns:")
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Pairplot for overall distribution and correlation analysis
print("\nGenerating Pairplot for Correlation Analysis...")
sns.pairplot(df[numerical_columns], diag_kind='kde', palette='viridis')
plt.show()

# Outlier detection using Z-score
def detect_outliers_zscore(data, column, threshold=3):
    """
    Detect outliers using the Z-score method.
    """
    mean_col = data[column].mean()
    std_col = data[column].std()
    z_scores = (data[column] - mean_col) / std_col
    return data[np.abs(z_scores) > threshold]

print("\nOutlier Detection Using Z-Score:")
outlier_counts = {}
for col in numerical_columns:
    outliers = detect_outliers_zscore(df, col)
    outlier_counts[col] = len(outliers)
    print(f"Outliers in {col}: {len(outliers)}")

# Removing outliers for all columns
def remove_outliers_zscore(data, threshold=3):
    """
    Remove outliers from all numerical columns using the Z-score method.
    """
    for col in numerical_columns:
        mean_col = data[col].mean()
        std_col = data[col].std()
        z_scores = (data[col] - mean_col) / std_col
        data = data[np.abs(z_scores) <= threshold]
    return data

df_cleaned = remove_outliers_zscore(df)
print(f"\nDataset cleaned. Remaining rows: {len(df_cleaned)}")

# Correlation analysis
correlation_matrix = df_cleaned.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap for correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Save the cleaned and analyzed dataset
df_cleaned.to_csv("fully_cleaned_global_warming_sim_dataset.csv", index=False)
print("\nFully cleaned and analyzed dataset saved as 'fully_cleaned_global_warming_sim_dataset.csv'.")
