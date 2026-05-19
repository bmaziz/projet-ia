import io
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pymongo import MongoClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

load_dotenv()

DATA_DIR = Path("data")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rag_medical")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TOP_K = 3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings_model = None
mongo_docs = None       # liste de dicts {text, embedding, metadata}
pdf_chunks = None       # liste de Documents LangChain avec embeddings
llm = None
prompt = None


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def vector_search(query_vector, docs, top_k):
    scored = [(cosine_similarity(query_vector, d["embedding"]), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def load_mongo_vectors():
    client = MongoClient(MONGO_URI)
    docs = list(client[MONGO_DB]["medicaments_vectors"].find(
        {}, {"_id": 1, "text": 1, "embedding": 1, "denomination": 1, "source": 1}
    ))
    client.close()
    return [
        {
            "text": d["text"],
            "embedding": d["embedding"],
            "metadata": {"source": "mongodb", "id": str(d["_id"]), "denomination": d.get("denomination", "")}
        }
        for d in docs
    ]


def load_pdf_vectors(embeddings_model):
    chunks_data = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        texts = [c.page_content for c in chunks]
        vectors = embeddings_model.embed_documents(texts)
        for chunk, vector in zip(chunks, vectors):
            chunks_data.append({
                "text": chunk.page_content,
                "embedding": vector,
                "metadata": chunk.metadata
            })
    return chunks_data


@app.on_event("startup")
async def startup():
    global embeddings_model, mongo_docs, pdf_chunks, llm, prompt

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY manquante.")

    print("Chargement du modèle d'embeddings...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Chargement des vecteurs depuis MongoDB...")
    mongo_docs = load_mongo_vectors()
    print(f"  {len(mongo_docs)} médicaments chargés")

    print("Indexation des PDFs...")
    pdf_chunks = load_pdf_vectors(embeddings_model)
    print(f"  {len(pdf_chunks)} chunks PDF indexés")

    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)

    prompt = PromptTemplate.from_template("""
Tu es un assistant médical spécialisé en pharmacologie.
Tu dois répondre uniquement à partir du contexte fourni.

Consignes :
- Réponds en français.
- je veux que les réponse soit plus intelligente et plus détaillé
- Si l'information est absente, dis : "Je ne trouve pas cette information dans les documents fournis."
- Réponse claire, structurée et concise.
- Termine par "Sources :" avec les fichiers utilisés.

Contexte :
{context}

Question :
{question}
""")
    print("Assistant RAG prêt.")


def retrieve(question):
    query_vector = embeddings_model.embed_query(question)
    all_docs = mongo_docs + pdf_chunks
    return vector_search(query_vector, all_docs, TOP_K)


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc["metadata"].get("source", "?")
        page = doc["metadata"].get("page", "")
        label = f"source={source}" + (f" | page={page + 1}" if isinstance(page, int) else "")
        parts.append(f"[Extrait {i} | {label}]\n{doc['text']}")
    return "\n\n".join(parts)


def format_sources(docs):
    seen = []
    for doc in docs:
        source = doc["metadata"].get("source", "?")
        page = doc["metadata"].get("page", "")
        item = source + (f" (page {page + 1})" if isinstance(page, int) else "")
        if item not in seen:
            seen.append(item)
    return ", ".join(seen)


COMPARE_KEYWORDS = ["compare", "comparer", "différence", "difference", "versus", "vs", "entre"]
INTERACTION_KEYWORDS = ["interaction", "dangereux", "danger", "prends", "prendre", "associer", "ensemble", "compatible", "incompatible"]
PDF_KEYWORDS = ["pdf", "génère", "genere", "générer", "fiche", "document", "télécharger", "telecharger"]


def find_med_in_text(q: str, meds: list):
    """Retourne tous les médicaments trouvés dans le texte."""
    found = []
    for med in meds:
        name = med.get("denomination", "").lower()
        substance = med.get("substance_active", "").lower()
        if (name and name.split()[0] in q) or (substance and substance in q):
            if med["_id"] not in [f["_id"] for f in found]:
                found.append(med)
    return found


def detect_compare_request(question: str):
    q = question.lower()
    if not any(k in q for k in COMPARE_KEYWORDS):
        return None
    client = MongoClient(MONGO_URI)
    meds = list(client[MONGO_DB]["medicaments_vectors"].find({}, {"denomination": 1, "substance_active": 1}))
    client.close()
    found = find_med_in_text(q, meds)
    return [m["_id"] for m in found[:2]] if len(found) >= 2 else None


def detect_interaction_request(question: str):
    q = question.lower()
    if not any(k in q for k in INTERACTION_KEYWORDS):
        return None
    client = MongoClient(MONGO_URI)
    meds = list(client[MONGO_DB]["medicaments_vectors"].find({}, {"denomination": 1, "substance_active": 1, "interactions": 1}))
    client.close()
    found = find_med_in_text(q, meds)
    if len(found) < 2:
        return None
    # Vérifier si l'un interagit avec l'autre
    alerts = []
    for i, med_a in enumerate(found):
        for med_b in found[i+1:]:
            name_b = med_b.get("denomination", "").lower()
            substance_b = med_b.get("substance_active", "").lower()
            interactions_a = [x.lower() for x in med_a.get("interactions", [])]
            if any(name_b.split()[0] in inter or substance_b in inter for inter in interactions_a):
                alerts.append({"med_a": med_a["denomination"], "med_b": med_b["denomination"], "dangerous": True})
            else:
                alerts.append({"med_a": med_a["denomination"], "med_b": med_b["denomination"], "dangerous": False})
    return alerts if alerts else None


def detect_pdf_request(question: str, last_med_id: str | None = None):
    q = question.lower()
    if not any(k in q for k in PDF_KEYWORDS):
        return None
    client = MongoClient(MONGO_URI)
    meds = list(client[MONGO_DB]["medicaments_vectors"].find({}, {"denomination": 1, "substance_active": 1}))
    client.close()
    for med in meds:
        name = med.get("denomination", "").lower()
        substance = med.get("substance_active", "").lower()
        if name and name.split()[0] in q:
            return med["_id"]
        if substance and substance.lower() in q:
            return med["_id"]
    return last_med_id


def generate_pdf(med_id: str) -> io.BytesIO:
    client = MongoClient(MONGO_URI)
    med = client[MONGO_DB]["medicaments_vectors"].find_one({"_id": med_id}, {"embedding": 0, "text": 0})
    client.close()

    if not med:
        raise ValueError(f"Médicament {med_id} introuvable")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#1a73e8"), spaceAfter=6)
    subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
                                     fontSize=12, textColor=colors.HexColor("#5f6368"), spaceAfter=16)
    section_style = ParagraphStyle("section", parent=styles["Heading2"],
                                    fontSize=12, textColor=colors.HexColor("#1a73e8"),
                                    spaceBefore=14, spaceAfter=4)
    body_style = ParagraphStyle("body", parent=styles["Normal"],
                                 fontSize=10, leading=16, spaceAfter=6)

    elements = []

    # En-tête
    elements.append(Paragraph(med.get("denomination", ""), title_style))
    elements.append(Paragraph(f"Substance active : {med.get('substance_active', '')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a73e8")))
    elements.append(Spacer(1, 12))

    # Tableau infos générales
    voies = ", ".join(med.get("voie_administration", []))
    table_data = [
        ["Forme", med.get("forme", "-")],
        ["Voie d'administration", voies or "-"],
        ["Statut AMM", med.get("statut_amm", "-")],
    ]
    table = Table(table_data, colWidths=[5*cm, 12*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0fe")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a73e8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dadce0")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # Sections
    sections = [
        ("Indications", med.get("indications", "-")),
        ("Posologie", med.get("posologie", "-")),
        ("Contre-indications", med.get("contre_indications", "-")),
        ("Effets indésirables", med.get("effets_indesirables", "-")),
        ("Interactions médicamenteuses", ", ".join(med.get("interactions", [])) or "-"),
    ]
    for title, content in sections:
        elements.append(Paragraph(title, section_style))
        elements.append(Paragraph(content, body_style))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dadce0")))
    elements.append(Paragraph("Document généré par Assistant Médical RAG", 
                               ParagraphStyle("footer", parent=styles["Normal"],
                                              fontSize=8, textColor=colors.grey)))

    doc.build(elements)
    buf.seek(0)
    return buf


class QuestionRequest(BaseModel):
    question: str
    last_med_id: str | None = None


@app.post("/ask")
async def ask(body: QuestionRequest):
    # Détection interaction dangereuse
    interaction_alerts = detect_interaction_request(body.question)
    if interaction_alerts is not None:
        docs = retrieve(body.question)
        context = format_context(docs)
        final_prompt = prompt.format(context=context, question=body.question)
        response = llm.invoke(final_prompt).content
        return {"answer": response, "interactions": interaction_alerts, "last_med_id": body.last_med_id}

    # Détection comparaison
    compare_ids = detect_compare_request(body.question)
    if compare_ids:
        return {"answer": "", "compare_ids": compare_ids}

    # Détection PDF
    med_id = detect_pdf_request(body.question, body.last_med_id)
    if med_id:
        return {"answer": "📄 Votre fiche PDF est prête !", "pdf_id": med_id}

    docs = retrieve(body.question)
    top_med_id = next(
        (d["metadata"]["id"] for d in docs if d["metadata"].get("source") == "mongodb"), None
    )
    context = format_context(docs)
    sources = format_sources(docs)
    final_prompt = prompt.format(context=context, question=body.question)
    response = llm.invoke(final_prompt).content
    if "Sources :" not in response:
        response = response.strip() + f"\n\nSources : {sources}"
    return {"answer": response, "sources": sources, "last_med_id": top_med_id}


@app.get("/compare")
async def compare(id1: str, id2: str):
    client = MongoClient(MONGO_URI)
    fields = {"embedding": 0, "text": 0}
    med1 = client[MONGO_DB]["medicaments_vectors"].find_one({"_id": id1}, fields)
    med2 = client[MONGO_DB]["medicaments_vectors"].find_one({"_id": id2}, fields)
    client.close()
    if not med1 or not med2:
        return {"error": "Médicament introuvable"}
    rows = ["denomination", "substance_active", "forme", "indications", "posologie", "contre_indications", "effets_indesirables"]
    return {"med1": {r: med1.get(r, "-") for r in rows}, "med2": {r: med2.get(r, "-") for r in rows}}


@app.get("/pdf/{med_id}")
async def download_pdf(med_id: str):
    buf = generate_pdf(med_id)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={med_id}.pdf"}
    )
