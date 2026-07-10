"""Pipeline stage 3 (RQ3): collect grounded answers with token logprobs for entropy analysis.

Reads: data/framed_prompts.csv. Writes: data/rag_responses_logprobs.jsonl.
"""
import os, json, time, math
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompts_full = pd.read_csv("data/framed_prompts.csv")

unique_article_ids = prompts_full["article_id"].drop_duplicates().head(300)
prompts = prompts_full[prompts_full["article_id"].isin(unique_article_ids)].copy()

OUTPUT_PATH = "data/rag_responses_logprobs.jsonl"
MODEL_NAME = "gpt-4o-mini"

RAG_PROMPT_TEMPLATE = """Answer the following question using only the information in the article below. Do not use any outside knowledge. If the article does not contain enough information to answer, say so explicitly.

Article:
{article_text}

Question: {question}

Answer in 1-2 sentences."""


def compute_mean_entropy(logprobs_content):
    """Compute mean token entropy from OpenAI's top-logprob output.
    NOTE: OpenAI exposes only the top-k logprobs (not the full distribution),
    so this is a partial-distribution entropy estimate, not true full-vocabulary
    entropy. This is stated explicitly as a limitation in the thesis."""
    if not logprobs_content:
        return None
    token_entropies = []
    for token_info in logprobs_content:
        top_logprobs = token_info.top_logprobs if hasattr(token_info, "top_logprobs") else []
        if not top_logprobs:
            continue
        probs = [math.exp(t.logprob) for t in top_logprobs]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        token_entropies.append(entropy)
    if not token_entropies:
        return None
    return sum(token_entropies) / len(token_entropies)


def query_openai(article_text, question):
    truncated_article = article_text[:3000]
    prompt = RAG_PROMPT_TEMPLATE.format(article_text=truncated_article, question=question)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
        logprobs=True,
        top_logprobs=5,
    )
    choice = resp.choices[0]
    answer = choice.message.content.strip()
    token_count = len(choice.logprobs.content) if choice.logprobs and choice.logprobs.content else 0
    mean_entropy = compute_mean_entropy(choice.logprobs.content) if choice.logprobs else None
    return answer, mean_entropy, token_count


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
print(f"Target: {len(prompts)} framed prompts (300 articles x 3 frames) on {MODEL_NAME}\n")

with open(OUTPUT_PATH, "a") as f:
    for _, row in prompts.iterrows():
        key = (row["article_id"], row["frame_type"])
        if key in completed:
            continue

        max_retries = 4
        for attempt in range(1, max_retries + 1):
            try:
                answer, mean_entropy, token_count = query_openai(str(row["text"]), row["framed_question"])
                record = {
                    "article_id": row["article_id"],
                    "bias_rating": row["bias_rating"],
                    "frame_type": row["frame_type"],
                    "model": MODEL_NAME,
                    "answer": answer,
                    "mean_entropy": mean_entropy,
                    "token_count": token_count,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"\u2713 article {row['article_id']} | {row['frame_type']} | entropy={mean_entropy}")
                time.sleep(0.1)
                break

            except Exception as e:
                err_text = str(e)
                is_rate_limit = "429" in err_text or "rate" in err_text.lower()
                wait = 15 if is_rate_limit else 2 * attempt
                print(f"ERROR (attempt {attempt}/{max_retries}): article {row['article_id']} | {e}")
                if attempt == max_retries:
                    print("  -> giving up on this call for now, will retry on next script run")
                else:
                    print(f"  -> waiting {wait}s before retry")
                    time.sleep(wait)

print(f"\nDone. Responses with entropy saved to {OUTPUT_PATH}")