import json
import pandas as pd

# Batch 2 case
labels2 = []
with open("data/hallucination_labels_batch2.jsonl") as f:
    for line in f:
        labels2.append(json.loads(line))
labels2_df = pd.DataFrame(labels2)
flagged2 = labels2_df[labels2_df["hallucination_label"] != "SUPPORTED"]

responses2 = []
with open("data/rag_responses_openai_batch2.jsonl") as f:
    for line in f:
        responses2.append(json.loads(line))
resp2_df = pd.DataFrame(responses2)

prompts2_df = pd.read_csv("data/framed_prompts_batch2.csv")
text_lookup2 = prompts2_df.drop_duplicates(subset=["article_id"]).set_index("article_id")["text"].to_dict()
q_lookup2 = prompts2_df.set_index(["article_id", "frame_type"])["framed_question"].to_dict()

print("=== BATCH 2 FLAGGED CASES ===\n")
for _, row in flagged2.iterrows():
    aid, frame = row["article_id"], row["frame_type"]
    ans_row = resp2_df[(resp2_df["article_id"]==aid) & (resp2_df["frame_type"]==frame)]
    answer = ans_row.iloc[0]["answer"] if not ans_row.empty else "NOT FOUND"
    print(f"ARTICLE {aid} | {frame} | {row['hallucination_label']}")
    print(f"Question: {q_lookup2.get((aid,frame),'?')}")
    print(f"Answer: {answer}")
    print(f"Article: {text_lookup2.get(aid,'?')}")
    print()

# Llama 3.1 cases
labels3 = []
with open("data/hallucination_labels_batch3_llama31.jsonl") as f:
    for line in f:
        labels3.append(json.loads(line))
labels3_df = pd.DataFrame(labels3)
flagged3 = labels3_df[labels3_df["hallucination_label"] != "SUPPORTED"]

responses3 = []
with open("data/rag_responses_llama31_batch3.jsonl") as f:
    for line in f:
        responses3.append(json.loads(line))
resp3_df = pd.DataFrame(responses3)

prompts3_df = pd.read_csv("data/framed_prompts_batch3.csv")
text_lookup3 = prompts3_df.drop_duplicates(subset=["article_id"]).set_index("article_id")["text"].to_dict()
q_lookup3 = prompts3_df.set_index(["article_id", "frame_type"])["framed_question"].to_dict()

print("=== LLAMA 3.1 FLAGGED CASES ===\n")
for _, row in flagged3.iterrows():
    aid, frame = row["article_id"], row["frame_type"]
    ans_row = resp3_df[(resp3_df["article_id"]==aid) & (resp3_df["frame_type"]==frame)]
    answer = ans_row.iloc[0]["answer"] if not ans_row.empty else "NOT FOUND"
    print(f"ARTICLE {aid} | {frame} | {row['hallucination_label']}")
    print(f"Question: {q_lookup3.get((aid,frame),'?')}")
    print(f"Answer: {answer}")
    print(f"Article: {text_lookup3.get(aid,'?')}")
    print()