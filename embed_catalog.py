import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

# Pfade zu deinen Dateien
CSV_PATH = "data/outfits.csv"
EMB_PATH = "data/outfit_embeddings.pt"

def main():
    # 1. Katalog einlesen
    df = pd.read_csv(CSV_PATH)

    print(f"{len(df)} Outfits im Katalog gefunden.")

    # 2. CLIP Modell + Processor laden
    model_name = "openai/clip-vit-base-patch32"
    print(f"Lade CLIP Modell: {model_name} ...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 3. Text-Beschreibung pro Outfit bauen
    #    (damit CLIP etwas Sinnvolles embedden kann)
    texts = [
        f"{row['title']}. Color: {row['color']}. Tags: {row['tags']}"
        for _, row in df.iterrows()
    ]

    print("Beispiele für Beschreibungen:")
    for t in texts[:3]:
        print("  -", t)

    # 4. Texte durch den Processor schicken
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # 5. Embeddings berechnen (ohne Gradienten, nur Inferenz)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # 6. Normalisieren (wichtig für Cosine Similarity)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 7. Embeddings speichern
    torch.save(text_embeds, EMB_PATH)
    print(f"Embeddings gespeichert in: {EMB_PATH}")

if __name__ == "__main__":
    main()
