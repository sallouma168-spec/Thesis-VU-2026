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
text_lookup = prompts_df.drop_duplicates(subset=["article_id"]).set_index("article_id")["text"].to_dict()
question_lookup = prompts_df.set_index(["article_id", "frame_type"])["framed_question"].to_dict()

flagged = labels_df[labels_df["hallucination_label"] != "SUPPORTED"]
print(f"Total flagged: {len(flagged)}\n")

for _, row in flagged.iterrows():
    aid, frame = row["article_id"], row["frame_type"]
    answer_row = responses_df[(responses_df["article_id"]==aid) & (responses_df["frame_type"]==frame)]
    answer = answer_row.iloc[0]["answer"] if not answer_row.empty else "NOT FOUND"
    article = text_lookup.get(aid, "NOT FOUND")
    question = question_lookup.get((aid, frame), "NOT FOUND")

    print(f"{'='*80}")
    print(f"ARTICLE {aid} | {frame} | LABEL: {row['hallucination_label']}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Article: {article}")
    print()