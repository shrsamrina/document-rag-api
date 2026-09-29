# Document RAG API

This project is a backend API I built for document ingestion and conversational RAG.

The API allows a user to upload PDF or TXT files, processes the text into chunks, creates embeddings, and stores them in Qdrant. Users can then ask questions about the uploaded documents through a chat endpoint.

I also added Redis conversation memory so the API can handle follow-up questions within the same session, as well as an interview booking flow that collects the user's name, email, preferred date, and time.

## How it works

For document ingestion, the flow is:

```text
PDF/TXT
   ↓
Text extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
```

I implemented two chunking options:

- `fixed` - splits the text into fixed-size chunks with overlap.
- `line` - groups lines together up to a maximum chunk size.

Document information such as the filename, chunking strategy, and number of chunks is stored in SQLite.

For questions, the flow is:

```text
Question
   ↓
Create question embedding
   ↓
Search Qdrant
   ↓
Retrieve relevant chunks
   ↓
Build prompt with context
   ↓
Gemini
   ↓
Answer
```

The RAG logic is implemented directly instead of using `RetrievalQAChain`.

If the answer cannot be found in the retrieved document context, the model is instructed to say that there is not enough information in the provided documents.

## Conversation memory

Redis is used to keep conversation history for each session.

This allows follow-up questions to use the previous conversation for understanding references while still using the uploaded documents as the source of information.

The Redis session expires after 24 hours.

## Interview booking

The `/chat` endpoint can also detect when a user wants to schedule an interview.

The user first confirms that they want to continue with the booking. The API then collects:

- Name
- Email
- Date
- Time

The temporary booking state is stored in Redis. Once all the required information is collected, the booking is saved in SQLite.

## Technologies used

- FastAPI
- Python
- Gemini
- Sentence Transformers
- Qdrant
- Redis
- SQLite
- SQLAlchemy
- Pydantic
- PyPDF

## Project structure

```text
document-rag-api/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── schemas.py
│   └── services.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`main.py` contains the API endpoints, while the `app` folder contains configuration, database models, schemas, and service logic.

## Running the project

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and provide the required values:

```env
GEMINI_API_KEY=
REDIS_HOST=
REDIS_PORT=
REDIS_USERNAME=
REDIS_PASSWORD=
```

Then start the API:

```bash
uvicorn main:app
```

The Swagger documentation can be opened at:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

### `POST /documents`

Uploads a PDF or TXT document.

The `strategy` parameter can be either `fixed` or `line`.

### `POST /chat`

Handles questions about the uploaded documents, follow-up conversations, and interview booking.

Example:

```json
{
  "session_id": "user-1",
  "question": "What programming languages does Samrina know?"
}
```

### `GET /history/{session_id}`

Returns the stored conversation history for a session.

## Notes

The project uses Qdrant as the vector database. It does not use FAISS or Chroma.

The RAG pipeline is custom-built and does not use `RetrievalQAChain`.

There is no frontend UI. FastAPI Swagger is only used to test and document the API.
