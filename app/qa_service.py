from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from app.config import CHROMA_DIR, MODEL_NAME, GROQ_API_KEY
from jinja2 import Template
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import os
import time
from pathlib import Path

# --- Custom Embedding class using local model ---
class LocalAlbertEmbeddings(Embeddings):
    def __init__(self):
        model_path = Path(__file__).parent / "local_models" / "paraphrase-albert-small-v2"
        self.model = SentenceTransformer(str(model_path.resolve()))

    def embed_documents(self, texts):
        return [vec.tolist() for vec in self.model.encode(texts)]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Set Groq API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Shared embedding instance
embeddings = LocalAlbertEmbeddings()

# --- Resolution Generator ---
def get_resolution(query: str):
    greeting_keywords = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    normalized_query = query.strip().lower()

    if any(keyword in normalized_query for keyword in greeting_keywords):
        return {
            "resolution": "Hi there! How can I help you today? Please describe the incident you're investigating.",
            "render_text": "👋 <strong>Hi there!</strong> How can I help you today? Please describe the incident you're investigating."
        }

    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return {
            "resolution": "No similar incidents found.",
            "render_text": "❌ No matching past incidents were found in the system."
        }

    doc, score = results[0]
    metadata = doc.metadata or {}
    description = doc.page_content or ""

    # Extract metadata with fallbacks
    meta = {
        "ticket_id": metadata.get("ticket_id", "N/A"),
        "assignee": metadata.get("assignee", "N/A"),
        "severity": metadata.get("severity", "unknown"),
        "cif": metadata.get("cif", "N/A"),
        "lcin": metadata.get("lcin", "N/A"),
        "account": metadata.get("account", "N/A"),
        "scheme_code": metadata.get("scheme_code", "N/A"),
        "timestamp": metadata.get("timestamp", "N/A"),
        "msguid": metadata.get("msguid", "N/A"),
        "mat_log": metadata.get("mat_log", "N/A"),
        "mat_error": metadata.get("mat_error", "N/A"),
        "source": metadata.get("source", "N/A"),
        "priority": metadata.get("priority", "N/A"),
        "impacted_service": metadata.get("impacted_service", "N/A"),
        "segment": metadata.get("segment", "N/A"),
        "channel": metadata.get("channel", "N/A"),
        "comments": metadata.get("comments", "")
    }

    # Prepare HTML comments
    if meta["comments"]:
        lines = meta["comments"].strip().split("\n")[:5]
        comments_html = "<ul>\n" + "\n".join(f"<li>{line}</li>" for line in lines) + "\n</ul>"
    else:
        comments_html = "<p>No comments were recorded for this incident.</p>"

    # Prompt to LLM
    prompt = (
        f"You are investigating a new issue reported by a customer:\n\n"
        f"{query}\n\n"
        f"A past similar incident was found:\n"
        f"{description[:2000]}\n\n"
        f"Based on the details, summarize the resolution that was applied earlier. "
        f"Be specific and explain what fixed the issue and who resolved it. Mention severity too."
    )

    llm = ChatGroq(model_name=MODEL_NAME, temperature=0)
    resolution = "Unable to generate resolution."

    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            resolution = response.content.strip()
            break
        except Exception as e:
            print(f"[Groq] Attempt {attempt + 1} failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    else:
        resolution = "⚠️ Resolution could not be generated due to system error."

    # Render clean summary with metadata
    render_template = Template("""
        ✅ This issue seems similar to a previous one (Ticket ID: <strong>{{ ticket_id }}</strong>) and was resolved by <strong>{{ assignee }}</strong>.
        <br /><br />
        🛠️ <strong>Resolution Summary:</strong><br />
        {{ resolution }}<br /><br />

        🔍 <strong>Incident Details:</strong><br />
        • <strong>Severity:</strong> {{ severity }}<br />
        • <strong>Priority:</strong> {{ priority }}<br />
        • <strong>Impacted Service:</strong> {{ impacted_service }}<br />
        • <strong>Business Segment:</strong> {{ segment }}<br />
        • <strong>Source:</strong> {{ source }}<br />
        • <strong>Channel:</strong> {{ channel }}<br /><br />

        🧾 <strong>Customer Identifiers:</strong><br />
        • <strong>CIF:</strong> {{ cif }}<br />
        • <strong>LCIN:</strong> {{ lcin }}<br />
        • <strong>Account Number:</strong> {{ account }}<br />
        • <strong>Scheme Code:</strong> {{ scheme_code }}<br /><br />

        🕓 <strong>Reported At:</strong> {{ timestamp }}<br />
        📄 <strong>Log:</strong> {{ mat_log }}<br />
        ⚠️ <strong>Error:</strong> {{ mat_error }}<br />
        📨 <strong>Message ID:</strong> {{ msguid }}<br /><br />

        💬 <strong>Resolution Comments:</strong><br />
        {{ comments_html | safe }}
    """)

    render_text = render_template.render(
        resolution=resolution,
        comments_html=comments_html,
        **meta
    )

    return {
        "resolution": resolution,
        "render_text": render_text,
        **meta
    }
