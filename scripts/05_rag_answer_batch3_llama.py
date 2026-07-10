"""Pipeline stage 3: grounded answering — query Llama 3.3 70B (Groq) on the framed prompts.

Reads: data/framed_prompts_batch3.csv. Writes: data/rag_responses_llama33_batch3.jsonl.
"""
import os, json, time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompts = pd.read_csv("data/framed_prompts_batch3.csv")

OUTPUT_PATH = "data/rag_responses_llama33_batch3.jsonl"
MODEL_NAME = "llama-3.3-70b-versatile"

RAG_PROMPT_TEMPLATE = """Answer the following question using only the information in the article below. Do not use any outside knowledge. If the article does not contain enough information to answer, say so explicitly.

Article:
{article_text}

Question: {question}

Answer in 1-2 sentences."""


def query_groq(article_text, question):
    truncated_article = article_text[:3000]
    prompt = RAG_PROMPT_TEMPLATE.format(article_text=truncated_article, question=question)
    resp = client.chat.completions.create(
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
print(f"Target: {len(prompts)} framed prompts on {MODEL_NAME}\n")
print("NOTE: this will likely hit Groq's daily token quota partway through.")
print("Just re-run this same command tomorrow (and the day after) to continue.\n")

with open(OUTPUT_PATH, "a") as f:
    for _, row in prompts.iterrows():
        key = (row["article_id"], row["frame_type"])
        if key in completed:
            continue

        max_retries = 4
        for attempt in range(1, max_retries + 1):
            try:
                answer = query_groq(str(row["text"]), row["framed_question"])
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
                time.sleep(0.5)
                break

            except Exception as e:
                err_text = str(e)
                is_rate_limit = "429" in err_text or "rate" in err_text.lower()
                wait = 30 if is_rate_limit else 2 * attempt
                print(f"ERROR (attempt {attempt}/{max_retries}): article {row['article_id']} | {e}")
                if attempt == max_retries:
                    print("  -> giving up on this call for now, will retry on next script run")
                else:
                    print(f"  -> waiting {wait}s before retry")
                    time.sleep(wait)

print(f"\nRun finished (or paused by daily quota). Responses saved to {OUTPUT_PATH}")
print("If paused, simply re-run this script tomorrow to continue from where it left off.")