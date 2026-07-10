import re, json
import pandas as pd

gt = pd.read_csv("data/ground_truth.csv")
gt_lookup = {row["fact_id"]: row for _, row in gt.iterrows()}

HEDGE_WORDS = [
    "approximately", "around", "about", "roughly",
    "i believe", "i think", "may have", "might have",
    "not certain", "unclear", "i'm not sure", "could be",
    "estimated", "i was unable to verify",
    "i cannot confirm", "i'm not entirely sure"
]

def extract_number(text):
    match = re.search(r'\b(\d{1,3}(?:\.\d{1,2})?)\b', text)
    return float(match.group(1)) if match else None

def is_hedged(text):
    t = text.lower()
    return any(w in t for w in HEDGE_WORDS)

def is_refused(text):
    t = text.lower()
    return any(w in t for w in [
        "i don't know", "cannot", "don't have",
        "no information", "i'm unable", "not available",
        "i do not have"
    ])

records = []
with open("data/raw_responses.jsonl") as f:
    for line in f:
        r = json.loads(line)
        fact_id = r["fact_id"]

        if fact_id not in gt_lookup:
            continue

        fact = gt_lookup[fact_id]
        response = r["response_text"]
        extracted = extract_number(response)
        correct_val = float(fact["correct_value"])
        hedged = is_hedged(response)
        refused = is_refused(response)

        if refused:
            verdict = "refused"
            error_direction = None
        elif extracted is None:
            verdict = "no_value"
            error_direction = None
        else:
            tolerance = 0.6 if fact["unit"] == "percent" else 0.4
            if abs(extracted - correct_val) <= tolerance:
                verdict = "correct"
                error_direction = None
            else:
                verdict = "error"
                error_direction = "over" if extracted > correct_val else "under"

        records.append({
            "prompt_id": r["prompt_id"],
            "fact_id": fact_id,
            "frame_type": r["frame_type"],
            "model": r["model"],
            "repetition": r["repetition"],
            "country": fact["country"],
            "party_name": fact["party_name"],
            "party_abbr": fact["party_abbr"],
            "fact_type": fact["fact_type"],
            "correct_value": correct_val,
            "extracted_value": extracted,
            "verdict": verdict,
            "hedged": hedged,
            "error_direction": error_direction,
            "response_text": response
        })

df = pd.DataFrame(records)
df.to_csv("data/coded_results.csv", index=False)

print("=== OVERALL VERDICTS ===")
print(df["verdict"].value_counts())
print("\n=== VERDICTS BY FRAME TYPE ===")
print(df.groupby("frame_type")["verdict"].value_counts().unstack(fill_value=0))
print("\n=== ERROR RATE BY FRAME TYPE ===")
df["is_error"] = (df["verdict"] == "error").astype(int)
print(df.groupby("frame_type")["is_error"].mean().round(3))
print(f"\nTotal rows coded: {len(df)}")
print("Saved to data/coded_results.csv")