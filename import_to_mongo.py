"""
Importe tous les fichiers JSON du dossier data/ dans MongoDB.
Usage : python import_to_mongo.py
"""
import json
from pathlib import Path
from pymongo import MongoClient, UpdateOne

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "rag_medical"
COLLECTION = "medicaments"
DATA_DIR = Path("data")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION]

ops = []
total = 0

for json_path in sorted(DATA_DIR.glob("*.json")):
    with open(json_path, encoding="utf-8") as f:
        medicaments = json.load(f)

    for med in medicaments:
        med.pop("embedding", None)  # supprimer les embeddings JSON, FAISS les recalcule
        ops.append(UpdateOne(
            {"_id": med["_id"]},
            {"$set": med},
            upsert=True
        ))
        total += 1

if ops:
    result = collection.bulk_write(ops)
    print(f"✓ {total} médicaments importés/mis à jour dans MongoDB")
    print(f"  Insérés : {result.upserted_count} | Modifiés : {result.modified_count}")
else:
    print("Aucun fichier JSON trouvé dans data/")

client.close()
