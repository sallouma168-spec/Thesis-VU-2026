"""Build the manual RQ2 directional-coding sheet from the flagged hallucination cases.

Reads: data/framed_prompts.csv, data/hallucination_labels.jsonl, data/rag_responses_openai.jsonl. Writes: data/coding_sheet.txt.
"""
import json
import pandas as pd

labels = []
with open("data/hallucination_labels.jsonl") as f:
    for line in f:
        labels.append(json.loads(line))
labels_df = pd.DataFrame(labels)

responses = []
with open("data/rag_responses_openai.jsonl") as f:
    for line in f:
        responses.append(json.loads(line))
responses_df = pd.DataFrame(responses)

prompts_df = pd.read_csv("data/framed_prompts.csv")

merged = labels_df.merge(
    responses_df[["article_id", "frame_type", "title", "source", "framed_question"]],
    on=["article_id", "frame_type"],
    how="left"
)
merged = merged.merge(
    prompts_df[["article_id", "text"]].drop_duplicates(subset=["article_id"]),
    on="article_id",
    how="left"
)

unsupported = merged[merged["hallucination_label"] == "UNSUPPORTED"].copy()
unsupported = unsupported.sort_values(["bias_rating", "frame_type", "article_id"])

print(f"Total unsupported cases: {len(unsupported)}\n")
print("=" * 100)

with open("data/coding_sheet.txt", "w") as out:
    for i, row in unsupported.iterrows():
        block = f"""
CASE {row['article_id']} | source lean: {row['bias_rating'].upper()} | frame: {row['frame_type'].upper()}
Title: {row['title']}
Outlet: {row['source']}

Question asked: {row['framed_question']}

Model's answer: {row['answer']}

Article excerpt (first 500 chars):
{str(row['text'])[:500]}

CODE HERE (left / right / neutral / unclear): ___________
NOTES: ___________________________________________________

{"=" * 100}
"""
        out.write(block)
        print(block)

print(f"\nFull coding sheet also saved to data/coding_sheet.txt")