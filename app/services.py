from sentence_transformers import SentenceTransformer

from app.schemas import BookingDetails
from app.config import client, redis_client


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_booking_details(message: str) -> BookingDetails:
    prompt = f"""
Extract interview booking information from the user's message.

User message:
{message}

Return only the information that is actually provided.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": BookingDetails,
        }
    )

    return response.parsed


def save_booking_state(session_id: str, details: BookingDetails):
    key = f"booking:{session_id}"

    redis_client.hset(
        key,
        mapping={
            "name": details.name or "",
            "email": str(details.email) if details.email else "",
            "date": details.date.isoformat() if details.date else "",
            "time": details.time.isoformat() if details.time else "",
            "confirmed": str(details.confirmed)
        }
    )

    redis_client.expire(key, 86400)


def get_booking_state(session_id: str) -> BookingDetails:
    key = f"booking:{session_id}"

    data = redis_client.hgetall(key)

    return BookingDetails(
        name=data.get("name") or None,
        email=data.get("email") or None,
        date=data.get("date") or None,
        time=data.get("time") or None,
        confirmed=data.get("confirmed") == "True"
    )


def merge_booking_details(
    old: BookingDetails,
    new: BookingDetails
) -> BookingDetails:

    return BookingDetails(
        name=new.name or old.name,
        email=new.email or old.email,
        date=new.date or old.date,
        time=new.time or old.time,
        confirmed=old.confirmed
    )


def is_booking_active(session_id: str) -> bool:
    key = f"booking:{session_id}"

    return redis_client.exists(key) == 1


def detect_booking_intent(message: str) -> bool:
    prompt = f"""
Determine whether the user wants to book or schedule an interview.

User message:
{message}

Answer only with:
yes
or
no
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text.strip().lower() == "yes"


# 1st chunking

def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# 2nd chunking

def line_based_chunking(
    text: str,
    max_chunk_size: int = 500
):
    chunks = []
    current_chunk = ""

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(current_chunk) + len(line) <= max_chunk_size:
            current_chunk += line + "\n"

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
