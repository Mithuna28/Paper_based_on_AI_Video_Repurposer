import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv


# =========================
# LOAD ENVIRONMENT
# =========================

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from .env"
    )


# =========================
# OPENROUTER CONFIGURATION
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Change this model if you have selected a different OpenRouter model.
MODEL = "openrouter/free"


def call_openrouter(prompt: str) -> str:

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "AI Video Repurposer",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected OpenRouter response: {data}"
        )

    if not content or not content.strip():
        raise ValueError("OpenRouter returned an empty response.")

    return content.strip()


# =========================
# AI TITLE GENERATION
# =========================

def generate_title(
    topic: str,
    captions: str
) -> str:

    prompt = f"""
Create one short, catchy title for a short-form video.

Topic:
{topic}

Selected captions:
{captions}

Rules:
- Maximum 80 characters
- Make it engaging and natural
- Clearly match the video topic
- Do not use quotation marks
- Return ONLY the title
"""

    text = call_openrouter(prompt)

    title = re.sub(
        r"^```(?:text)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE
    )

    title = re.sub(
        r"^(?:[-*]\s+|\d+[.)]\s+|title:\s*)",
        "",
        title,
        flags=re.IGNORECASE
    )

    title = title.splitlines()[0].strip().strip("\"'`* ")

    if not title:
        raise ValueError(
            "OpenRouter returned an empty title."
        )

    return title[:80]


# =========================
# AI HASHTAG GENERATION
# =========================

def generate_hashtags(
    topic: str,
    captions: str
) -> list[str]:

    prompt = f"""
Generate 8 relevant hashtags for a short-form video.

Topic:
{topic}

Selected captions:
{captions}

Rules:
- Return exactly 8 hashtags
- Each must start with #
- No spaces inside a hashtag
- Relevant to the topic and content
- Mix broad and specific hashtags
- Return ONLY the hashtags separated by spaces
"""

    text = call_openrouter(prompt)

    text = re.sub(
        r"```(?:text)?|```",
        "",
        text,
        flags=re.IGNORECASE
    )

    hashtags = []
    seen = set()

    for match in re.finditer(
        r"#([A-Za-z0-9_]+)",
        text
    ):
        hashtag = f"#{match.group(1)}"
        normalized = hashtag.casefold()

        if normalized not in seen:
            hashtags.append(hashtag)
            seen.add(normalized)

        if len(hashtags) == 8:
            break

    if len(hashtags) != 8:
        raise ValueError(
            f"OpenRouter returned only "
            f"{len(hashtags)} unique hashtags; "
            f"expected 8."
        )

    return hashtags