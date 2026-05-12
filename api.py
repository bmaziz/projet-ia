import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()

DATA_DIR = Path("data")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "rag_medical")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
llm = None
prompt = None


def load_pdf_documents(data_dir: Path):
    documents = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
    return documents


def load_mongo_documents():
    client = MongoClient(MONGO_URI)
    medicaments = list(client[MONGO_DB]["medicaments"].find({}, {"embedding": 0}))
    client.close()
    documents = []
    for med in medicaments:
        interactions = ", ".join(med.get("interactions", []))
        voies = ", ".join(med.get("voie_administration", []))
        content = (
            f"Médicament : {med.get('denomination', '')}\n"
            f"Substance active : {med.get('substance_active', '')}\n"
            f"Forme : {med.get('forme', '')}\n"
            f"Voie d'administration : {voies}\n"
            f"Statut AMM : {med.get('statut_amm', '')}\n"
            f"Indications : {med.get('indications', '')}\n"
            f"Contre-indications : {med.get('contre_indications', '')}\n"
            f"Effets indésirables : {med.get('effets_indesirables', '')}\n"
            f"Posologie : {med.get('posologie', '')}\n"
            f"Interactions : {interactions}"
        )
        documents.append(Document(
            page_content=content,
            metadata={"source": "mongodb", "id": str(med.get("_id", ""))}
        ))
    return documents


def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        parts.append(f"[Extrait {i} | source={source} | page={page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(docs):
    seen = []
    for doc in docs:
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        item = f"{source} (page {page})"
        if item not in seen:
            seen.append(item)
    return ", ".join(seen)


@app.on_event("startup")
async def startup():
    global retriever, llm, prompt

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY manquante.")

    documents = load_pdf_documents(DATA_DIR) + load_mongo_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)

    template = """
Tu es un assistant médical spécialisé en pharmacologie.
Tu dois répondre uniquement à partir du contexte fourni.

Consignes :
- Réponds en français.
- Si l'information est absente, dis : "Je ne trouve pas cette information dans les documents fournis."
- Réponse claire, structurée et concise.
- Termine par "Sources :" avec les fichiers utilisés.

Contexte :
{context}

Question :
{question}
"""
    prompt = PromptTemplate.from_template(template)


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(body: QuestionRequest):
    docs = retriever.invoke(body.question)
    context = format_context(docs)
    sources = format_sources(docs)
    final_prompt = prompt.format(context=context, question=body.question)
    response = llm.invoke(final_prompt).content
    if "Sources :" not in response:
        response = response.strip() + f"\n\nSources : {sources}"
    return {"answer": response, "sources": sources}
