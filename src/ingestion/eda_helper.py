import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_class_distribution(df: pd.DataFrame):
    counts = df['Label'].value_counts()
    plt.figure(figsize=(14, 5))
    counts.plot(kind='bar', color='steelblue')
    plt.title('Class Distribution — CICIDS 2017')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    Path('docs').mkdir(exist_ok=True)
    plt.savefig('docs/class_distribution.png', dpi=150)
    plt.show()
    print(counts)

def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include='number').iloc[:, :20]
    corr = numeric.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap='coolwarm', center=0, fmt='.1f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('docs/correlation_heatmap.png', dpi=150)
    plt.show()

def find_high_correlation_features(df: pd.DataFrame, threshold=0.95) -> list:
    numeric = df.select_dtypes(include='number')
    corr = numeric.corr().abs()
    to_drop = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] > threshold:
                to_drop.add(corr.columns[i])
    print(f'Features to drop (correlation > {threshold}): {len(to_drop)}')
    print(to_drop)
    return list(to_drop)