import pandas as pd

df = pd.read_csv("data/allsides_full.csv")
print(f"Columns: {list(df.columns)}")

# check for a date column under various likely names
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
print(f"Possible date columns: {date_cols}")

if date_cols:
    print(df[date_cols[0]].describe())
    print(df[date_cols[0]].head())