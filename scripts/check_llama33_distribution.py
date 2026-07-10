import json
from collections import Counter

records = []
with open("data/hallucination_labels_batch3_llama33.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total labeled: {len(records)}")
print(f"\nLabel distribution by frame:")
for frame in ["neutral", "positive", "negative"]:
    frame_records = [r for r in records if r["frame_type"] == frame]
    counts = Counter(r["hallucination_label"] for r in frame_records)
    total = len(frame_records)
    supported = counts.get("SUPPORTED", 0)
    unsupported = counts.get("UNSUPPORTED", 0)
    contradictory = counts.get("CONTRADICTORY", 0)
    rate = (unsupported + contradictory) / total * 100 if total else 0
    print(f"  {frame} (n={total}): supported={supported}, unsupported={unsupported}, contradictory={contradictory} -> {rate:.2f}% hallucination rate")