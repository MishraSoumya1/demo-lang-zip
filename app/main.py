from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import csv
import io
import re

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

from app.models import IncidentQuery
from app.qa_service import get_resolution
from app.config import CHROMA_DIR
from pathlib import Path

# --- Custom Embeddings using SentenceTransformer ---
class LocalAlbertEmbeddings(Embeddings):
    def __init__(self):
        model_path = Path(__file__).parent / "local_models" / "paraphrase-albert-small-v2"
        self.model = SentenceTransformer(str(model_path.resolve()))

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.encode(texts)]

    def embed_query(self, text):
        return self.model.encode(text).tolist()


# --- FastAPI App Init ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Query Endpoint ---
@app.post("/query")
def query_incident(payload: IncidentQuery):
    resolution = get_resolution(payload.query)
    return {"resolution": resolution}


# --- Ingestion Endpoint ---
@app.post("/ingest")
async def ingest_incidents(
    authorization: Optional[str] = Header(None),
    file: Optional[UploadFile] = File(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        reader = csv.reader(io.StringIO(decoded), delimiter="\t")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    docs = []
    line_number = 0
    for row in reader:
        line_number += 1
        if len(row) < 3:
            continue  # skip malformed lines

        ticket_id = row[0].strip()
        date = row[1].strip()
        message = row[2].strip()

        # Extract Issue Description from message
        description_match = re.search(r"\|Issue Description\|(.+?)\|", message)
        if not description_match:
            continue  # skip if Issue Description not found

        description = description_match.group(1).strip()

        # Extract key-value metadata pairs
        parts = message.split("|")
        metadata = {}
        i = 1
        while i < len(parts) - 1:
            raw_key = parts[i].strip()
            raw_val = parts[i + 1].strip()
            if raw_key and raw_val and raw_key != "Issue Description":
                normalized_key = raw_key.lower().replace(" ", "_")
                metadata[normalized_key] = raw_val
            i += 2

        # Include ticket_id and date explicitly
        metadata["ticket_id"] = ticket_id
        metadata["date"] = date

        docs.append(Document(page_content=description, metadata=metadata))

    if not docs:
        raise HTTPException(status_code=400, detail="No valid documents found in file.")

    # Store in Chroma vector DB
    try:
        embeddings = LocalAlbertEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB error: {str(e)}")

    return JSONResponse(content={"status": "success", "count": len(docs)}, status_code=200)