import json
import pandas as pd

prompts = pd.read_csv("data/framed_prompts_batch3.csv")
expected_keys = set(zip(prompts["article_id"], prompts["frame_type"]))

records = []
with open("data/rag_responses_llama31_batch3.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

actual_keys = set((r["article_id"], r["frame_type"]) for r in records)

missing = expected_keys - actual_keys
print(f"Expected: {len(expected_keys)}")
print(f"Actual: {len(actual_keys)}")
print(f"Missing: {len(missing)}")
if missing:
    print("Missing entries:", sorted(missing))

seen = set()
dupes = 0
for r in records:
    key = (r["article_id"], r["frame_type"])
    if key in seen:
        dupes += 1
    seen.add(key)
print(f"Duplicates: {dupes}")