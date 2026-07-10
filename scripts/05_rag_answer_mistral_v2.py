"""Pipeline stage 3: grounded answering — query Mistral (EXCLUDED, exploratory) on the framed prompts.

Reads: data/framed_prompts.csv. Writes: data/rag_responses_mistral_v2.jsonl.
"""
import os, json, time
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

prompts_df = pd.read_csv("data/framed_prompts.csv")

OUTPUT_PATH = "data/rag_responses_mistral_v2.jsonl"
MODEL_NAME = "mistral-large-latest"

SECONDS_BETWEEN_CALLS = 0.25

RAG_PROMPT_TEMPLATE = """Answer the following question using only the information in the article below. Do not use any outside knowledge. If the article does not contain enough information to answer, say so explicitly.

Article:
{article_text}

Question: {question}

Answer in 1-2 sentences."""


def query_mistral(article_text, question):
    truncated_article = article_text[:3000]
    prompt = RAG_PROMPT_TEMPLATE.format(article_text=truncated_article, question=question)
    resp = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def already_done():
    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec["article_id"], rec["frame_type"])
                    done.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


completed = already_done()
print(f"Found {len(completed)} responses already collected. Resuming...\n")
print(f"Target: {len(prompts_df)} framed prompts on {MODEL_NAME}\n")
print(f"Pacing requests at ~{1/SECONDS_BETWEEN_CALLS:.0f}/second to stay under the 5 req/sec free-tier limit\n")

with open(OUTPUT_PATH, "a") as f:
    for _, row in prompts_df.iterrows():
        key = (row["article_id"], row["frame_type"])
        if key in completed:
            continue

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                answer = query_mistral(str(row["text"]), row["framed_question"])
                record = {
                    "article_id": row["article_id"],
                    "title": row["title"],
                    "source": row["source"],
                    "bias_rating": row["bias_rating"],
                    "frame_type": row["frame_type"],
                    "framed_question": row["framed_question"],
                    "model": MODEL_NAME,
                    "answer": answer,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"\u2713 article {row['article_id']} | {row['frame_type']} | {row['bias_rating']}")
                time.sleep(SECONDS_BETWEEN_CALLS)
                break

            except Exception as e:
                err_text = str(e)
                is_rate_limit = "429" in err_text or "rate" in err_text.lower()
                wait = 5 * attempt if is_rate_limit else 2 * attempt
                print(f"ERROR (attempt {attempt}/{max_retries}): article {row['article_id']} | {e}")
                if attempt == max_retries:
                    print("  -> giving up on this call for now, will retry on next script run")
                else:
                    print(f"  -> waiting {wait}s before retry")
                    time.sleep(wait)

print(f"\nDone. Responses saved to {OUTPUT_PATH}")