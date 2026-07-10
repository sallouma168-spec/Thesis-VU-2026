import json
from collections import Counter

records = []
with open("data/hallucination_labels_batch2.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total labeled: {len(records)}")
print(f"\nLabel distribution overall:")
print(Counter(r["hallucination_label"] for r in records))

print(f"\nLabel distribution by frame:")
for frame in ["neutral", "positive", "negative"]:
    frame_records = [r for r in records if r["frame_type"] == frame]
    counts = Counter(r["hallucination_label"] for r in frame_records)
    total = len(frame_records)
    supported_pct = counts.get("SUPPORTED", 0) / total * 100 if total else 0
    print(f"  {frame} (n={total}): {dict(counts)} -> {supported_pct:.1f}% supported")