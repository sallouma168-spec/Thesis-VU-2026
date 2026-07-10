import pandas as pd

df = pd.read_csv("data/allsides_full.csv")

# Stricter check: does the article actually discuss the EUROPEAN PARLIAMENT ELECTION process
# (not just mention an EU-adjacent figure or Brexit in passing)
strict_keywords = ["european parliament election", "eu parliament election",
                    "meps", "european parliament seats", "european elections"]

mask = df["text"].fillna("").str.lower().apply(
    lambda t: any(kw in t for kw in strict_keywords)
)
strict_matches = df[mask]
print(f"Articles specifically about EU Parliament elections: {len(strict_matches)}")
if len(strict_matches) > 0:
    for _, row in strict_matches.head(10).iterrows():
        print(f"\n{row['title']} ({row['bias_rating']})")
        print(row['text'][:200])