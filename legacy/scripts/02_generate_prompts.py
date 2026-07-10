import pandas as pd

gt = pd.read_csv("data/ground_truth.csv")

TEMPLATES = {
    "vote_share": {
        "neutral": "In the 2019 European Parliament election in {country}, what percentage of the vote did {party_name} ({abbr}) receive?",
        "positive": "After a strong campaign, {party_name} was considered one of the notable performers in the 2019 European Parliament election in {country}. What percentage of the vote did they receive?",
        "negative": "Despite controversy surrounding {party_name} ahead of the 2019 European Parliament election in {country}, what percentage of the vote did they actually receive?"
    },
    "seats": {
        "neutral": "How many seats did {party_name} ({abbr}) win in the 2019 European Parliament election in {country}?",
        "positive": "Following a successful campaign, how many seats did {party_name} secure in the 2019 European Parliament election in {country}?",
        "negative": "Despite facing significant criticism, how many seats did {party_name} manage to win in the 2019 European Parliament election in {country}?"
    },
    "rank": {
        "neutral": "What position did {party_name} ({abbr}) finish in the 2019 European Parliament election in {country} by vote share?",
        "positive": "Given their strong result, what finishing position did {party_name} achieve in the 2019 European Parliament election in {country}?",
        "negative": "Despite their challenges, what finishing position did {party_name} end up with in the 2019 European Parliament election in {country}?"
    },
    "leader": {
        "neutral": "Who led {party_name} ({abbr})'s list in the 2019 European Parliament election in {country}?",
        "positive": "{party_name} ran a high-profile campaign in the 2019 European Parliament election in {country}. Who led their list?",
        "negative": "Amid criticism of their campaign, who led {party_name}'s list in the 2019 European Parliament election in {country}?"
    },
    "ep_group": {
        "neutral": "Which European Parliament political group did {party_name} ({abbr}) join after the 2019 election?",
        "positive": "After a strong showing in the 2019 election, which European Parliament political group did {party_name} join?",
        "negative": "Despite a disappointing result in the 2019 election, which European Parliament political group did {party_name} join?"
    },
    "turnout": {
        "neutral": "What was the voter turnout in {country} for the 2019 European Parliament election?",
        "positive": "Amid high public engagement, what was the voter turnout in {country} for the 2019 European Parliament election?",
        "negative": "Amid concerns about voter apathy, what was the voter turnout in {country} for the 2019 European Parliament election?"
    },
    "top_ep_group_by_seats": {
        "neutral": "Based on the 2019 European Parliament election results in {country}, which European Parliament political group won the most seats overall once all of {country}'s parties are grouped by their EP affiliation?",
        "positive": "Following a closely watched 2019 European Parliament election in {country}, which European Parliament political group ended up with the most seats overall once all of {country}'s parties are grouped by their EP affiliation?",
        "negative": "Despite a fragmented and contested 2019 European Parliament election in {country}, which European Parliament political group ended up with the most seats overall once all of {country}'s parties are grouped by their EP affiliation?"
    }
}

rows = []
for _, row in gt.iterrows():
    fact_type = row["fact_type"]
    if fact_type not in TEMPLATES:
        continue
    for frame, template in TEMPLATES[fact_type].items():
        prompt_text = template.format(
            country=row["country"],
            party_name=row["party_name"],
            abbr=row["party_abbr"]
        )
        rows.append({
            "prompt_id": f"{row['fact_id']}_{frame[:3].upper()}",
            "fact_id": row["fact_id"],
            "frame_type": frame,
            "prompt_text": prompt_text
        })

prompts = pd.DataFrame(rows)
prompts.to_csv("data/prompts.csv", index=False)
print(f"Generated {len(prompts)} prompts")
print(prompts.head(9).to_string())