import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from fastapi.middleware.cors import CORSMiddleware

# Pfade zu CSV und Embeddings
CSV_PATH = "data/outfits.csv"
EMB_PATH = "data/outfit_embeddings.pt"

# ---- Request-Body für unsere API ----

class RecommendRequest(BaseModel):
    query: str   # der Suchtext, z.B. "black minimal streetwear blazer"
    top_k: int = 3  # wie viele Outfits zurückgeben

# ---- FastAPI App anlegen ----

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js Dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Daten & Modell beim Start laden ----

df = pd.read_csv(CSV_PATH)
catalog_embeds = torch.load(EMB_PATH)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

print(f"{len(df)} Outfits geladen. Embeddings aus {EMB_PATH} geladen.")

def embed_text(query: str) -> torch.Tensor:
    """
    Nimmt einen Text (z.B. 'black minimal blazer') und gibt
    einen normalisierten Embedding-Vektor als Torch Tensor zurück.
    """
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        text_emb = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # Normalisieren (für Cosine Similarity)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb[0]

# ---- Endpoint: /recommend ----

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Nimmt einen Suchtext und gibt die Top-K Outfits zurück.
    """

    # 1. Query-Text embedden
    q_emb = embed_text(req.query)

    # 2. Cosine-Ähnlichkeit mit allen Katalog-Embeddings
    sims = (catalog_embeds @ q_emb)

    # 3. Top-K Indizes holen
    topk_vals, topk_idx = torch.topk(sims, k=min(req.top_k, len(df)))

    results = []
    for score, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
        row = df.iloc[idx]
        results.append({
            "id": int(row["id"]),
            "title": row["title"],
            "image_url": row["image_url"],
            "color": row["color"],
            "tags": row["tags"],
            "score": float(score),
        })

    return {"query": req.query, "results": results}
