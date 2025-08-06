import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import CHROMA_DIR

# Load mock incident data
with open("mock_incidents.json", "r") as f:
    incidents = json.load(f)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert incidents into langchain Documents
docs = []
for incident in incidents:
    # Join all comments into a single block
    comment_text = "\n".join(
        [f"{comment['author']}: {comment['body']}" for comment in incident.get("comments", [])]
    )

    # Construct combined content for vector embedding
    content = (
        f"Title: {incident['title']}\n"
        f"Description: {incident['description']}\n"
        f"Comments:\n{comment_text}\n"
        f"Fix: {incident['fix_summary']}\n"
    )

    metadata = {
        "ticket_id": incident["ticket_id"],
        "assignee": incident["assignee"],
        "severity": incident["severity"]
    }

    docs.append(Document(page_content=content, metadata=metadata))

# Store in Chroma vector DB
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
vectorstore.persist()

print("✅ Mock incidents with comment arrays ingested into Chroma vector store.")
