import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3


def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    return documents


def load_json_documents(data_dir: Path):
    json_paths = sorted(data_dir.glob("*.json"))
    documents = []

    for json_path in json_paths:
        with open(json_path, encoding="utf-8") as f:
            medicaments = json.load(f)

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
                metadata={"source": json_path.name, "id": med.get("_id", "")}
            ))

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_prompt():
    template = """
Tu es un assistant medical specialise en pharmacologie.
Tu dois repondre uniquement a partir du contexte fourni.

Consignes importantes :
- Reponds en francais.
- Si l’information n’est pas presente dans le contexte, dis clairement :
"Je ne trouve pas cette information dans les documents fournis."
- Donne une reponse claire, structuree et concise.
- Termine par une ligne "Sources :" avec les fichiers utilises.

Contexte :
{context}

Question :
{question}
"""
    return PromptTemplate.from_template(template)


def format_context(docs):
    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")

        if isinstance(page, int):
            page = page + 1

        parts.append(
            f"[Extrait {i} | source={source} | page={page}]\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def format_sources(docs):
    unique_sources = []

    for doc in docs:
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")

        if isinstance(page, int):
            page = page + 1

        item = f"{source} (page {page})"
        if item not in unique_sources:
            unique_sources.append(item)

    return ", ".join(unique_sources)


def answer_question(question, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = format_context(docs)
    sources = format_sources(docs)

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt).content

    if "Sources :" not in response:
        response = response.strip() + f"\n\nSources : {sources}"

    return response, docs


def main():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "La variable GROQ_API_KEY est absente."
        )

    print("Chargement des documents...")
    documents = load_pdf_documents(DATA_DIR) + load_json_documents(DATA_DIR)
    print(f"Nombre total de documents charges : {len(documents)}")

    print("Decoupage en chunks...")
    chunks = split_documents(documents)
    print(f"Nombre total de chunks : {len(chunks)}")

    print("Creation des embeddings et de l’index FAISS...")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )

    prompt = build_prompt()

    print("\nAssistant RAG pret.")
    print("Tapez votre question ou 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()

        if not question:
            print("Veuillez saisir une question.\n")
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Fin du programme.")
            break

        answer, docs = answer_question(question, retriever, llm, prompt)

        print("\n--- Reponse ---")
        print(answer)
"""
        print("\n--- Chunks recuperes ---")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "source_inconnue")
            page = doc.metadata.get("page", "?")

            if isinstance(page, int):
                page = page + 1

            preview = doc.page_content[:250].replace("\n", " ")
            print(f"{i}. {source} | page {page}")
            print(f" {preview}...")

        print()
"""

if __name__ == "__main__":
    main()