"""Pipeline stage 1: generate one neutral factual question per article.

Reads: data/sampled_articles_batch2.csv. Writes: data/articles_with_questions_batch2.csv.
"""
import os, json, time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

articles = pd.read_csv("data/sampled_articles_batch2.csv")

OUTPUT_PATH = "data/articles_with_questions_batch2.csv"
QUESTION_GEN_MODEL = "gpt-4o-mini"

QUESTION_PROMPT_TEMPLATE = """Read the following news article and generate exactly one clear, neutral question that captures the core factual claim of the article. The question should be answerable directly from the article's content. Do not include any opinion or framing in the question itself. Respond with only the question, nothing else.

Article:
{article_text}"""


def generate_question(article_text):
    truncated = article_text[:3000]
    prompt = QUESTION_PROMPT_TEMPLATE.format(article_text=truncated)
    resp = client.chat.completions.create(
        model=QUESTION_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def already_done():
    done = set()
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done = set(existing["article_id"].tolist())
    return done


# same offset logic as before: batch2 article_ids run 1000+
articles = articles.reset_index().rename(columns={"index": "article_id"})
articles["article_id"] = articles["article_id"] + 1000

completed_ids = already_done()
print(f"Found {len(completed_ids)} questions already generated (from Groq + any prior runs). Resuming with OpenAI...\n")

results = []
if os.path.exists(OUTPUT_PATH):
    results = pd.read_csv(OUTPUT_PATH).to_dict("records")

remaining = articles[~articles["article_id"].isin(completed_ids)]
print(f"Remaining to generate: {len(remaining)}\n")

for _, row in remaining.iterrows():
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            question = generate_question(str(row["text"]))
            results.append({
                "article_id": row["article_id"],
                "title": row["title"],
                "source": row["source"],
                "bias_rating": row["bias_rating"],
                "text": row["text"],
                "generated_question": question,
                "question_gen_model": "gpt-4o-mini",  # tag so it's clear which model generated this
            })
            print(f"\u2713 article {row['article_id']} ({row['bias_rating']}): {question[:80]}")

            if len(results) % 25 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

            time.sleep(0.1)
            break

        except Exception as e:
            err_text = str(e)
            is_rate_limit = "429" in err_text or "rate" in err_text.lower()
            wait = 15 if is_rate_limit else 2 * attempt
            print(f"ERROR (attempt {attempt}/{max_retries}): article {row['article_id']} | {e}")
            if attempt == max_retries:
                print("  -> giving up on this article for now, will retry on next script run")
            else:
                print(f"  -> waiting {wait}s before retry")
                time.sleep(wait)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. {len(results)} total questions saved to {OUTPUT_PATH}")