import json
import pandas as pd

prompts = pd.read_csv("data/prompts.csv")
REPETITIONS = 5

expected = set()
for _, row in prompts.iterrows():
    for rep in range(1, REPETITIONS + 1):
        expected.add((row["prompt_id"], "gpt-4o-mini", rep))

actual = set()
with open("data/raw_responses_openai_logprobs.jsonl") as f:
    for line in f:
        try:
            rec = json.loads(line)
            actual.add((rec["prompt_id"], rec["model"], rec["repetition"]))
        except (json.JSONDecodeError, KeyError):
            continue

missing = expected - actual
print(f"Expected: {len(expected)}")
print(f"Actual:   {len(actual)}")
print(f"Missing:  {len(missing)}")
print()
if missing:
    print("Missing entries:")
    for m in sorted(missing):
        print(f"  {m}")