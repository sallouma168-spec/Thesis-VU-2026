"""
Samples a balanced subset of election articles for the RAG pipeline run.
Balances across bias_rating (left/center/right) as evenly as the data allows.

Usage: python scripts/02_sample_articles.py
"""
import pandas as pd

df = pd.read_csv("data/election_articles.csv")

TARGET_TOTAL = 500
RANDOM_SEED = 42  # fixed seed so the sample is reproducible

bias_counts = df["bias_rating"].value_counts()
print(f"Available election articles by lean:\n{bias_counts}\n")

# even split target, capped by whatever category has the fewest available
n_categories = df["bias_rating"].nunique()
target_per_category = TARGET_TOTAL // n_categories

samples = []
for category in df["bias_rating"].unique():
    available = df[df["bias_rating"] == category]
    n = min(target_per_category, len(available))
    sample = available.sample(n=n, random_state=RANDOM_SEED)
    samples.append(sample)
    print(f"{category}: sampled {n} of {len(available)} available")

sampled_df = pd.concat(samples).reset_index(drop=True)
print(f"\nTotal sampled: {len(sampled_df)}")
print(f"Final distribution:\n{sampled_df['bias_rating'].value_counts()}")

sampled_df.to_csv("data/sampled_articles.csv", index=False)
print(f"\nSaved to data/sampled_articles.csv")