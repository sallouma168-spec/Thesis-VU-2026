"""Judge validation: run the ORIGINAL (buggy) judge that misclassified honest hedging as UNSUPPORTED.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT_TEMPLATE = """You are checking whether an AI-generated answer is fully supported by a news article. The answer may contain multiple separate claims, even if it reads as one sentence. Check EVERY individual claim in the answer separately against the article, not just the main gist.

Classify the answer into exactly one category:

- SUPPORTED: every single claim in the answer, including minor or incidental details, is directly stated in or clearly implied by the article. If even one added detail is not in the article, this does not qualify as SUPPORTED.
- UNSUPPORTED: the answer's main claim may be correct, but it adds at least one detail, fact, or elaboration that the article does not state at all (regardless of whether that detail happens to be true in the real world).
- CONTRADICTORY: the answer states something that directly conflicts with what the article says.

If the answer mixes a correct core claim with one unsupported added detail, classify it as UNSUPPORTED, not SUPPORTED.

Respond with only one word: SUPPORTED, UNSUPPORTED, or CONTRADICTORY.

Article:
{article_text}

Question: {question}

Answer to check: {answer}"""

article = "Former New York Gov. George Pataki announced he is seeking the Republican presidential nomination in the 2016 election, joining a crowded field of candidates."
question = "What position is former New York Gov. George Pataki seeking in the 2016 election?"

test_answers = {
    "correct (should be SUPPORTED)": "George Pataki is seeking the Republican presidential nomination.",
    "contradictory (should be CONTRADICTORY)": "George Pataki is seeking the Democratic presidential nomination.",
    "unsupported, made-up detail (should be UNSUPPORTED)": "George Pataki is seeking the Republican presidential nomination, having served two terms as governor and previously worked as a senator.",
    "wrong office entirely (should be CONTRADICTORY or UNSUPPORTED)": "George Pataki is running for Senate in New York.",
}

for label, answer in test_answers.items():
    prompt = JUDGE_PROMPT_TEMPLATE.format(article_text=article, question=question, answer=answer)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    result = resp.choices[0].message.content.strip()
    print(f"{label}:")
    print(f"  Answer tested: {answer}")
    print(f"  Judge said: {result}\n")