# Assistant Médical RAG

## Lancement

### 1. Backend (FastAPI)
```bash
cd projet-ia
venv/bin/uvicorn api:app --reload
```
API disponible sur http://localhost:8000

### 2. Frontend (React + Vite)
```bash
cd projet-ia/frontend
npm install
npm run dev
```
Interface disponible sur http://localhost:5173

## Structure
```
projet-ia/
├── api.py              # API FastAPI (backend RAG)
├── rag_pdf_groq.py     # Script CLI original
├── data/               # Fichiers JSON et PDF
├── venv/               # Environnement Python
└── frontend/           # Interface React
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── main.jsx
    │   └── index.css
    ├── index.html
    ├── vite.config.js
    └── package.json
```
