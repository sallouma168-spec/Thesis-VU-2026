import json
import pandas as pd

labels = []
with open("data/hallucination_labels_deepseek_batch2.jsonl") as f:
    for line in f:
        labels.append(json.loads(line))
labels_df = pd.DataFrame(labels)
flagged = labels_df[labels_df["hallucination_label"] != "SUPPORTED"]

responses = []
with open("data/rag_responses_deepseek_batch2.jsonl") as f:
    for line in f:
        responses.append(json.loads(line))
resp_df = pd.DataFrame(responses)

prompts_df = pd.read_csv("data/framed_prompts_batch2.csv")
text_lookup = prompts_df.drop_duplicates(subset=["article_id"]).set_index("article_id")["text"].to_dict()
q_lookup = prompts_df.set_index(["article_id", "frame_type"])["framed_question"].to_dict()

print(f"Total flagged: {len(flagged)}\n")
for _, row in flagged.iterrows():
    aid, frame = row["article_id"], row["frame_type"]
    ans_row = resp_df[(resp_df["article_id"]==aid) & (resp_df["frame_type"]==frame)]
    answer = ans_row.iloc[0]["answer"] if not ans_row.empty else "NOT FOUND"
    print(f"ARTICLE {aid} | {frame} | {row['hallucination_label']}")
    print(f"Question: {q_lookup.get((aid,frame),'?')}")
    print(f"Answer: {answer}")
    print(f"Article: {text_lookup.get(aid,'?')}")
    print()