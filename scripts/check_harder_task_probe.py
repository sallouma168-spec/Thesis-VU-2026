import json
from collections import Counter

records = []
with open("data/harder_task_probe_results.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total responses: {len(records)}")
print(f"\nOverall distribution:")
print(Counter(r["hallucination_label"] for r in records))

print(f"\nBy frame:")
for frame in ["neutral", "positive", "negative"]:
    frame_records = [r for r in records if r["frame_type"] == frame]
    counts = Counter(r["hallucination_label"] for r in frame_records)
    total = len(frame_records)
    supported = counts.get("SUPPORTED", 0)
    rate = (total - supported) / total * 100 if total else 0
    print(f"  {frame} (n={total}): {dict(counts)} -> {rate:.1f}% hallucination rate")

print(f"\nFlagged cases:")
for r in records:
    if r["hallucination_label"] != "SUPPORTED":
        print(f"  doc {r['doc_id']} | {r['frame_type']} | {r['hallucination_label']}")
        print(f"    Q: {r['framed_question']}")
        print(f"    A: {r['answer']}")