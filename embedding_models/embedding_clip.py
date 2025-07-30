from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel

# ─── Load CLIP once ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ─── Embedding helpers ───────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Compute a single text embedding via CLIP, normalized to unit length."""
    if not text:
        return []
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
    # Normalize
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    return text_emb.cpu().numpy().flatten().tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute CLIP embeddings for a list of texts."""
    if not texts:
        return []
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embs = clip_model.get_text_features(**inputs)
    # Normalize along each row
    text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
    return text_embs.cpu().numpy().tolist()

# ─── Similarity search ───────────────────────────────────────────────────────────
def find_matches(query_embedding: list[float],
                 corpus_embeddings: list[list[float]],
                 threshold: float = 0.1):
    """
    Finds matches for a query embedding within a list of corpus embeddings
    using cosine similarity.
    """
    if not query_embedding or not corpus_embeddings:
        return []

    sims = cosine_similarity([query_embedding], corpus_embeddings)[0]
    print(f"Similarity scores: {sims}")
    matches = [(i, float(s)) for i, s in enumerate(sims)] #if s >= threshold]
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def get_prompt(crop_name):
    return f"A top-down view of a healthy {crop_name} plant at an early to middle growth stage, with visible leaves and structure"

def get_similar_classes_clip(candidates_classes: list[str],
                    target_class: str):
    """
    Compares classes of candidates with a query embedding and returns matches.
    """

    prompt = get_prompt(target_class)
    query_embedding = get_embedding(prompt)

    if not query_embedding:
        #print("No embedding found for the query.")
        return []

    cropus_prompts = [get_prompt(crop) for crop in candidates_classes]
    corpus_embeddings = get_embeddings(cropus_prompts)
    if not corpus_embeddings:
        #print("No embeddings found for the corpus.")
        return []
    
    matches = find_matches(query_embedding, corpus_embeddings, threshold= 0.2)    
    if not matches:
        #print("No matches found.")
        return []
    
    print(matches)
    matches.sort(key=lambda x: x[1], reverse=True)
    #print(f"Matches for '{target_class}':")
    #for idx, score in matches:
    #    print(f"- [{idx}] {candidates_classes[idx]} (sim={score:.4f})")

    #return the most closest match and any match whtin 8% similarity of it

    closest_match = matches[0]
    closest_match_score = closest_match[1]
    closest_match_index = closest_match[0]
    closest_match_class = candidates_classes[closest_match_index]
    #print(f"Closest match: {closest_match_class} (sim={closest_match_score:.4f})")

    # Find any match within 5% similarity of the closest match

    similar_matches = []
    for idx, score in matches[1:]:
        if score >= closest_match_score - 0.08:
            similar_matches.append(candidates_classes[idx])

    return [closest_match_class] + similar_matches

    