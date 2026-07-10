"""
Filters the full QBias CSV down to election-relevant articles.
Confirmed columns: 'title', 'tags', 'heading', 'source', 'text', 'bias_rating'

Usage: python scripts/01_filter_election_articles.py
"""
import pandas as pd

df = pd.read_csv("data/allsides_full.csv")

print(f"Full dataset: {len(df)} articles")
print(f"Bias distribution overall:\n{df['bias_rating'].value_counts()}\n")

# Election-relevant keywords to search in the 'tags' column
ELECTION_KEYWORDS = [
    "election", "elections", "voting", "voter", "ballot", "campaign",
    "presidential", "congress", "senate", "midterm", "primary", "polls"
]

mask = df["tags"].fillna("").str.lower().apply(
    lambda tags: any(kw in tags for kw in ELECTION_KEYWORDS)
)
election_df = df[mask].copy()

print(f"Election-relevant articles found: {len(election_df)}")
print(f"\nBias distribution among election articles:")
print(election_df["bias_rating"].value_counts())

election_df.to_csv("data/election_articles.csv", index=False)
print(f"\nSaved to data/election_articles.csv")