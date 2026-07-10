"""Draw a balanced article sample across left/right/center bias categories.

Reads: data/election_articles.csv, data/sampled_articles.csv. Writes: data/sampled_articles_batch2.csv.
"""
import pandas as pd

election_df = pd.read_csv("data/election_articles.csv")
used_df = pd.read_csv("data/sampled_articles.csv")

print(f"Total election articles available: {len(election_df)}")
print(f"Already used: {len(used_df)}")

used_keys = set(zip(used_df["title"], used_df["source"]))
election_df["key"] = list(zip(election_df["title"], election_df["source"]))
remaining_df = election_df[~election_df["key"].isin(used_keys)].drop(columns=["key"])

print(f"Remaining unused articles: {len(remaining_df)}")

TARGET_NEW = 502
RANDOM_SEED = 123

n_categories = remaining_df["bias_rating"].nunique()
target_per_category = TARGET_NEW // n_categories

samples = []
for category in remaining_df["bias_rating"].unique():
    available = remaining_df[remaining_df["bias_rating"] == category]
    n = min(target_per_category, len(available))
    sample = available.sample(n=n, random_state=RANDOM_SEED)
    samples.append(sample)
    print(f"{category}: sampled {n} of {len(available)} available")

new_sample_df = pd.concat(samples).reset_index(drop=True)
print(f"\nTotal newly sampled: {len(new_sample_df)}")
print(f"Distribution:\n{new_sample_df['bias_rating'].value_counts()}")

new_sample_df.to_csv("data/sampled_articles_batch2.csv", index=False)
print(f"\nSaved to data/sampled_articles_batch2.csv")