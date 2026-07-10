"""Pipeline stage 2: wrap each question in the neutral/positive/negative framing templates.

Reads: data/articles_with_questions_batch2.csv. Writes: data/framed_prompts_batch2.csv.
"""
import pandas as pd

df = pd.read_csv("data/articles_with_questions_batch2.csv")

FRAME_WRAPPERS = {
    "neutral": "{question}",
    "positive": "Following a notable and well-regarded development, {question_lower}",
    "negative": "Despite ongoing controversy surrounding this story, {question_lower}",
}

def lowercase_first_letter(s):
    if not s:
        return s
    return s[0].lower() + s[1:]

rows = []
for _, row in df.iterrows():
    question = str(row["generated_question"]).strip()
    question_lower = lowercase_first_letter(question)
    for frame, wrapper in FRAME_WRAPPERS.items():
        framed_question = wrapper.format(question=question, question_lower=question_lower)
        rows.append({
            "article_id": row["article_id"],
            "title": row["title"],
            "source": row["source"],
            "bias_rating": row["bias_rating"],
            "text": row["text"],
            "generated_question": question,
            "frame_type": frame,
            "framed_question": framed_question,
        })

framed_df = pd.DataFrame(rows)
framed_df.to_csv("data/framed_prompts_batch2.csv", index=False)
print(f"Generated {len(framed_df)} framed prompts from {len(df)} articles x 3 frames")