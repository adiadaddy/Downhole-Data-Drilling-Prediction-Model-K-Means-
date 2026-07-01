import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = r"C:\Users\admin\Downloads\archive (1)\log.csv"
data = pd.read_csv(path)

print("--- Data Diagnostic ---")
print(f"Original dataset size: {len(data)} rows")
print("Columns found in your CSV:", list(data.columns))

features = ['SP', 'RHOB', 'RILD', 'RLL3', 'RxoRt', 'MN', 'MI']
data_cleaned = data.dropna(subset=features).copy()
print(f"Dataset size after removing missing values: {len(data_cleaned)} rows\n")

X = data_cleaned[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("--- Running K-Means Clustering ---")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data_cleaned['Cluster_ID'] = kmeans.fit_predict(X_scaled)

print("Clustering successful!")
print(data_cleaned[['Depth', 'Cluster_ID']].head(20))

output_path = r"C:\Users\admin\Downloads\archive (1)\clustered_log_results.csv"
data_cleaned.to_csv(output_path, index=False)
print(f"\nResults saved successfully to: {output_path}")

print("\n--- GEOLOGICAL CLUSTER PROFILES ---")

cluster_profiles = data_cleaned.groupby('Cluster_ID')[features].mean()

print(cluster_profiles.round(2))

print("\n--- Meters of each rock type drilled ---")

cluster_counts = data_cleaned['Cluster_ID'].value_counts()
for cid, count in cluster_counts.items():
    print(f"Cluster {cid}: {count} rows ({count * 0.5} meters total)")
