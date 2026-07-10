import json
from collections import Counter

records = []
with open("data/hallucination_labels_batch2.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Before dedup: {len(records)}")

# keep only the FIRST occurrence of each (article_id, frame_type)
seen = set()
deduped = []
for r in records:
    key = (r["article_id"], r["frame_type"])
    if key not in seen:
        seen.add(key)
        deduped.append(r)

print(f"After dedup: {len(deduped)}")

# overwrite the file with clean, deduplicated data
with open("data/hallucination_labels_batch2.jsonl", "w") as f:
    for r in deduped:
        f.write(json.dumps(r) + "\n")

print("\nClean label distribution by frame:")
for frame in ["neutral", "positive", "negative"]:
    frame_records = [r for r in deduped if r["frame_type"] == frame]
    counts = Counter(r["hallucination_label"] for r in frame_records)
    total = len(frame_records)
    supported = counts.get("SUPPORTED", 0)
    unsupported = counts.get("UNSUPPORTED", 0)
    contradictory = counts.get("CONTRADICTORY", 0)
    halluc_rate = (unsupported + contradictory) / total * 100 if total else 0
    print(f"  {frame} (n={total}): supported={supported}, unsupported={unsupported}, contradictory={contradictory} -> {halluc_rate:.1f}% hallucination rate")