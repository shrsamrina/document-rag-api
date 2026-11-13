
import logging
import re
import time

from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException
from google.genai import errors
from sentence_transformers import SentenceTransformer

from app.schemas import BookingDetails
from app.config import client, redis_client


logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

GEMINI_MODEL = "gemini-3.5-flash-lite"


def generate_with_retry(
    *,
    model: str,
    contents: str,
    config: Optional[dict[str, Any]] = None,
    max_attempts: int = 3,
) -> Any:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(max_attempts):
        try:
            arguments = {
                "model": model,
                "contents": contents,
            }

            if config is not None:
                arguments["config"] = config

            return client.models.generate_content(**arguments)

        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)

            if status_code not in {429, 500, 502, 503, 504}:
                logger.error(
                    "Gemini API error: %s",
                    status_code,
                )
                raise HTTPException(
                    status_code=502,
                    detail="Gemini API request failed.",
                ) from exc

            if attempt == max_attempts - 1:
                logger.error(
                    "Gemini unavailable after %s attempts.",
                    max_attempts,
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Gemini is temporarily unavailable. "
                        "Please try again later."
                    ),
                ) from exc

            time.sleep(2 ** (attempt + 1))

    raise HTTPException(
        status_code=503,
        detail="Gemini service is unavailable.",
    )


def extract_booking_details(message: str) -> BookingDetails:
    prompt = f"""
Extract interview booking information from the user's message.

User message:
{message}

Rules:
- Extract only information explicitly provided.
- Never invent missing information.
- Convert times to 24-hour format.
- 8am means 08:00:00.
- 8pm means 20:00:00.
- 12am means 00:00:00.
- 12pm means 12:00:00.
- Preserve the user's AM/PM designation exactly.
- Do not assume UTC or add timezone information unless
  the user explicitly provides a timezone.
- Return dates in YYYY-MM-DD format.
- Do not invent a date, time, name, or email.
- Set confirmed to false.

Return structured booking details.
"""

    response = generate_with_retry(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": BookingDetails,
        },
    )

    if response.parsed is None:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return valid booking details.",
        )

    details = response.parsed

    time_match = re.search(
        r"\b(1[0-2]|0?[1-9])\s*(a\.?m\.?|p\.?m\.?)\b",
        message,
        flags=re.IGNORECASE,
    )

    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(2).lower().replace(".", "")

        if period == "am":
            hour = hour % 12
        else:
            hour = (hour % 12) + 12

        minute = 0

        details.time = datetime.strptime(
            f"{hour:02d}:{minute:02d}",
            "%H:%M",
        ).time()

    return details


def save_booking_state(
    session_id: str,
    details: BookingDetails,
) -> None:
    key = f"booking:{session_id}"

    redis_client.hset(
        key,
        mapping={
            "name": details.name or "",
            "email": str(details.email) if details.email else "",
            "date": (
                details.date.isoformat()
                if details.date
                else ""
            ),
            "time": (
                details.time.isoformat()
                if details.time
                else ""
            ),
            "confirmed": str(details.confirmed),
        },
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
        confirmed=data.get("confirmed") == "True",
    )


def merge_booking_details(
    old: BookingDetails,
    new: BookingDetails,
) -> BookingDetails:
    return BookingDetails(
        name=new.name or old.name,
        email=new.email or old.email,
        date=new.date or old.date,
        time=new.time or old.time,
        confirmed=old.confirmed,
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

    response = generate_with_retry(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response.",
        )

    return response.text.strip().lower() == "yes"


def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


def line_based_chunking(
    text: str,
    max_chunk_size: int = 500,
) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError(
            "max_chunk_size must be greater than zero"
        )

    chunks = []
    current_chunk = ""

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            for start in range(
                0,
                len(line),
                max_chunk_size,
            ):
                chunks.append(
                    line[start:start + max_chunk_size]
                )

            continue

        if (
            len(current_chunk)
            + len(line)
            + 1
            <= max_chunk_size
        ):
            current_chunk += line + "\n"

        else:
            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
