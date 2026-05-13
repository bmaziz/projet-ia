"""
Importe les médicaments dans MongoDB avec leurs vecteurs.
Usage : python import_to_mongo.py
"""
import json
from pathlib import Path
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_medical"
COLLECTION = "medicaments_vectors"
DATA_DIR = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_text(med):
    interactions = ", ".join(med.get("interactions", []))
    voies = ", ".join(med.get("voie_administration", []))
    return (
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


def main():
    # Charger tous les médicaments depuis les JSON
    medicaments = []
    for json_path in sorted(DATA_DIR.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for med in data:
            med.pop("embedding", None)
            medicaments.append(med)
    print(f"✓ {len(medicaments)} médicaments chargés depuis les fichiers JSON")

    # Calculer les embeddings
    print("Calcul des embeddings...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    texts = [build_text(med) for med in medicaments]
    vectors = embeddings_model.embed_documents(texts)
    print(f"✓ {len(vectors)} vecteurs calculés ({len(vectors[0])} dimensions)")

    # Stocker dans MongoDB
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]
    collection.drop()  # Réinitialiser la collection

    docs = []
    for med, text, vector in zip(medicaments, texts, vectors):
        docs.append({
            "_id": med["_id"],
            "denomination": med.get("denomination", ""),
            "substance_active": med.get("substance_active", ""),
            "forme": med.get("forme", ""),
            "voie_administration": med.get("voie_administration", []),
            "statut_amm": med.get("statut_amm", ""),
            "indications": med.get("indications", ""),
            "contre_indications": med.get("contre_indications", ""),
            "effets_indesirables": med.get("effets_indesirables", ""),
            "posologie": med.get("posologie", ""),
            "interactions": med.get("interactions", []),
            "text": text,
            "embedding": vector,
        })

    collection.insert_many(docs)
    print(f"✓ {len(docs)} documents stockés dans MongoDB (collection: {COLLECTION})")
    print(f"  Chaque document contient : données + texte + vecteur ({len(vectors[0])} dimensions)")

    client.close()


if __name__ == "__main__":
    main()
