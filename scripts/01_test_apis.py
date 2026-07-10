"""Smoke-test that every model API key and endpoint responds.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

resp = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    temperature=0
)

print("Groq response:", resp.choices[0].message.content)