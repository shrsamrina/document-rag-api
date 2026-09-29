from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from pypdf import PdfReader
from typing import Literal
from qdrant_client.models import PointStruct
from uuid import uuid4
import time

from app.database import SessionLocal, Document, Booking
from app.schemas import ChatRequest, BookingDetails
from app.config import client, redis_client, qdrant_client, COLLECTION_NAME
from app.services import (
    embedding_model,
    extract_booking_details,
    save_booking_state,
    get_booking_state,
    merge_booking_details,
    is_booking_active,
    detect_booking_intent,
    fixed_size_chunking,
    line_based_chunking
)


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Document RAG API is running"}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    strategy: Literal["fixed", "line"] = "fixed"
):

    allowed_types = ["application/pdf", "text/plain"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    # this is unique ID for this uploaded document
    document_id = str(uuid4())

    contents = await file.read()

    if file.content_type == "application/pdf":
        reader = PdfReader(BytesIO(contents))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    else:
        text = contents.decode("utf-8")

    if strategy == "fixed":
        chunks = fixed_size_chunking(text)

    elif strategy == "line":
        chunks = line_based_chunking(text)

    else:
        raise HTTPException(
            status_code=400,
            detail="Strategy must be 'fixed' or 'line'"
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from the document"
        )

    embeddings = embedding_model.encode(chunks)

    points = []

    for index, chunk in enumerate(chunks):
        point = PointStruct(
            id=str(uuid4()),

            # for conversion from numpy format
            vector=embeddings[index].tolist(),

            payload={
                "document_id": document_id,
                "text": chunk,
                "filename": file.filename,
                "strategy": strategy
            }
        )

        points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    db = SessionLocal()

    document = Document(
        id=document_id,
        filename=file.filename,
        content_type=file.content_type,
        strategy=strategy,
        chunk_count=len(chunks)
    )

    db.add(document)
    db.commit()
    db.close()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
        "stored_in_qdrant": len(points)
    }


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question

    booking_active = is_booking_active(request.session_id)

    if booking_active:
        details = get_booking_state(request.session_id)

        if not details.confirmed:
            answer = question.strip().lower()

            if answer in ["y", "yes"]:
                details.confirmed = True
                save_booking_state(request.session_id, details)

                return {"message": "What is your name?"}

            if answer in [
                "n",
                "no",
                "cancel",
                "stop",
                "never mind",
                "nevermind"
            ]:
                redis_client.delete(f"booking:{request.session_id}")

                return {
                    "message": "Interview booking cancelled."
                }

            return {
                "message": "Please answer Y or N. Would you like to continue with the interview booking?"
            }

        if question.strip().lower() in [
            "cancel",
            "stop",
            "never mind",
            "nevermind"
        ]:
            redis_client.delete(f"booking:{request.session_id}")

            return {
                "message": "Interview booking cancelled."
            }

        old_details = details
        new_details = extract_booking_details(question)

        details = merge_booking_details(
            old_details,
            new_details
        )

        save_booking_state(request.session_id, details)

        # what if details not provided situations
        if not details.name:
            return {"message": "What is your name?"}

        if not details.email:
            return {"message": "What is your email?"}

        if not details.date:
            return {
                "message": "What date would you like the interview?"
            }

        if not details.time:
            return {
                "message": "What time would you like the interview?"
            }

        db = SessionLocal()

        booking = Booking(
            id=str(uuid4()),
            name=details.name,
            email=str(details.email),
            date=details.date.isoformat(),
            time=details.time.isoformat()
        )

        db.add(booking)
        db.commit()
        db.close()

        redis_client.delete(f"booking:{request.session_id}")

        return {
            "message": "Interview booked successfully.",
            "name": details.name,
            "email": details.email,
            "date": details.date,
            "time": details.time
        }

    booking_intent = detect_booking_intent(question)

    if booking_intent:
        details = BookingDetails()

        save_booking_state(
            request.session_id,
            details
        )

        return {
            "message": "Would you like to continue with the interview booking? Please answer Y or N."
        }

    # RAG code
    history_key = f"chat:{request.session_id}"

    # for redis conversation memory
    history = redis_client.lrange(
        history_key,
        0,
        -1
    )

    history_text = "\n".join(history)

    # creating embedding for the question
    question_embedding = embedding_model.encode(
        question
    )

    # searching Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_embedding.tolist(),
        limit=3
    ).points

    # building context from retrieved chunks
    context = "\n\n".join(
        result.payload["text"]
        for result in search_results
    )

    # building RAG prompt
    prompt = f"""
Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I don't have enough information in the provided documents to answer that."

Use the conversation history only to understand references such as
"he", "she", "it", or previous questions.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    # calling Gemini with retries
    response = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            break

        except Exception:
            if attempt == 2:
                raise

            time.sleep(2)

    answer = response.text

    # saving conversation to Redis
    redis_client.rpush(
        history_key,
        f"User: {question}"
    )

    redis_client.rpush(
        history_key,
        f"Assistant: {answer}"
    )

    # this will expire the session after 24 hours
    redis_client.expire(
        history_key,
        86400
    )

    return {
        "question": question,
        "answer": answer
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    history_key = f"chat:{session_id}"

    history = redis_client.lrange(
        history_key,
        0,
        -1
    )

    return {
        "session_id": session_id,
        "history": history
    }
