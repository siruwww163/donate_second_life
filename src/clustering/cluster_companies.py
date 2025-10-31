"""
cluster_companies.py
---------------------------------
Performs company clustering for the Second Life e.V. donor outreach project.
Uses cleaned_contacts.csv from the previous processing step.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.environ["OMP_NUM_THREADS"] = "1"


# 1. Read processed data
file_path = "data/processed/cleaned_contacts.csv"
df = pd.read_csv(file_path)

# 2. Select relevant features
features = df[[
    "type", "outreach_type", "location",
    "is_contacted", "has_contact_info", "is_english"
]].copy()

# 3. Filling missing values
features[["type", "outreach_type", "location"]] = (
    features[["type", "outreach_type", "location"]].fillna("unknown")
)

# 4. One-hot code
features_encoded = pd.get_dummies(features, drop_first=True)

# 5. Normalizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_encoded)

# 6. Run KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# 7. Evaluating clustering quality
score = silhouette_score(X_scaled, df["cluster"])
print(f"\nClustering completed, a total of {df['cluster'].nunique()} groups")
print(f"Silhouette Score: {score:.3f}")

# 8. Basic information of each group
cluster_summary = df.groupby("cluster")[["is_contacted", "has_contact_info", "is_english"]].mean()
print("\n=== Cluster Summary (mean feature) ===")
print(cluster_summary)

# 9. Save data with cluster labels
output_path = "data/processed/clustered_contacts.csv"
df.to_csv(output_path, index=False)
print(f"\nClustering results have been saved to: {output_path}")

# 10. Preview representative companies in each category
print("\n=== Representative Company Examples ===")
for i in range(df["cluster"].nunique()):
    examples = df[df["cluster"] == i]["name"].head(3).tolist()
    print(f"Cluster {i}: {examples}")
