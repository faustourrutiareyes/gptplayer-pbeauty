import argparse
import json
import asyncio
from openai import AsyncOpenAI

async def query_openai(client, message):
    prompt = f"""
    You are given the following message: {message}
    Return a JSON object with two fields:
    - "number": a number you think fits the message
    - "COT": a short reasoning why you chose this number
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"number": None, "COT": text}

async def pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", default="output.json", help="Path to output JSON file")
    parser.add_argument("--api-key", required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    # Load messages
    with open(args.file, "r") as f:
        messages = json.load(f)

    # Create client with user's API key
    client = AsyncOpenAI(api_key=args.api_key)

    # Run all tasks concurrently
    tasks = [query_openai(client, msg) for msg in messages]
    results = await asyncio.gather(*tasks)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(messages)} messages. Output saved to {args.output}")
