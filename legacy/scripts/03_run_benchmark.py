import os, json, time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompts = pd.read_csv("data/prompts.csv")

REPETITIONS = 5
OUTPUT_PATH = "data/raw_responses.jsonl"

MODELS = [
    "llama-3.3-70b-versatile",
    "mistral-saba-24b",
]


def query_groq(prompt_text, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def already_done():
    """Read existing output file so we can resume without repeating work."""
    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (rec["prompt_id"], rec["model"], rec["repetition"])
                    done.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


completed = already_done()
print(f"Found {len(completed)} responses already collected. Resuming...\n")

total_calls = len(prompts) * len(MODELS) * REPETITIONS
print(f"Target: {len(prompts)} prompts x {len(MODELS)} models x {REPETITIONS} reps = {total_calls} calls\n")

with open(OUTPUT_PATH, "a") as f:
    for model_name in MODELS:
        for _, row in prompts.iterrows():
            for rep in range(1, REPETITIONS + 1):
                key = (row["prompt_id"], model_name, rep)
                if key in completed:
                    continue  # already collected, skip

                max_retries = 4
                for attempt in range(1, max_retries + 1):
                    try:
                        response_text = query_groq(row["prompt_text"], model_name)
                        record = {
                            "prompt_id": row["prompt_id"],
                            "fact_id": row["fact_id"],
                            "frame_type": row["frame_type"],
                            "model": model_name,
                            "repetition": rep,
                            "response_text": response_text,
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        print(f"\u2713 {model_name} | {row['prompt_id']} rep{rep} | {row['frame_type']}")
                        time.sleep(0.5)
                        break  # success, stop retrying

                    except Exception as e:
                        err_text = str(e)
                        is_rate_limit = "429" in err_text or "rate" in err_text.lower()
                        wait = 30 if is_rate_limit else 2 * attempt
                        print(f"ERROR (attempt {attempt}/{max_retries}): {model_name} | {row['prompt_id']} rep{rep} | {e}")
                        if attempt == max_retries:
                            print(f"  -> giving up on this call for now, will retry on next script run")
                        else:
                            print(f"  -> waiting {wait}s before retry")
                            time.sleep(wait)

print("\nRun finished (or paused by rate limit). Re-run this script later to pick up where it left off.")