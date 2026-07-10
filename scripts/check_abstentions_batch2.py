import json
import pandas as pd

labels = []
with open("data/hallucination_labels_batch2.jsonl") as f:
    for line in f:
        labels.append(json.loads(line))
labels_df = pd.DataFrame(labels)

flagged = labels_df[labels_df["hallucination_label"].isin(["UNSUPPORTED", "CONTRADICTORY"])]
print(f"Total unsupported/contradictory in batch 2: {len(flagged)}")

abstention_phrases = [
    "does not specify", "does not contain", "not enough information",
    "cannot determine", "does not provide", "does not mention",
    "does not state", "is not mentioned", "article does not"
]

abstentions_by_frame = {"neutral": 0, "positive": 0, "negative": 0}
for _, row in flagged.iterrows():
    answer = str(row["answer"]).lower()
    if any(phrase in answer for phrase in abstention_phrases):
        abstentions_by_frame[row["frame_type"]] += 1

print(f"Abstentions found: {abstentions_by_frame}")
print(f"Total abstentions: {sum(abstentions_by_frame.values())}")