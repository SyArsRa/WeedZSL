from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), 'imagebind'))
from imagebind.imagebind.models import imagebind_model
from imagebind.imagebind.models.imagebind_model import ModalityType
from imagebind.imagebind import data

# ─── Load ImageBind ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
bind_model = imagebind_model.imagebind_huge(pretrained=True)
bind_model.eval()
bind_model = bind_model.to(device)

# ─── Embedding helpers ────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Compute a single text embedding via ImageBind, normalized to unit length."""
    if not text:
        return []
    with torch.no_grad():
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], device),
        }
        emb = bind_model(inputs)[ModalityType.TEXT]
        emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().flatten().tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute ImageBind embeddings for a list of texts."""
    if not texts:
        return []
    with torch.no_grad():
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(texts, device),
        }
        embs = bind_model(inputs)[ModalityType.TEXT]
        embs = F.normalize(embs, p=2, dim=-1)
    return embs.cpu().numpy().tolist()

# ─── Similarity search ────────────────────────────────────────────────────────────
def find_matches(query_embedding: list[float],
                 corpus_embeddings: list[list[float]],
                 threshold: float = 0.1):
    if not query_embedding or not corpus_embeddings:
        return []

    sims = cosine_similarity([query_embedding], corpus_embeddings)[0]
    print(f"Similarity scores: {sims}")
    matches = [(i, float(s)) for i, s in enumerate(sims)]
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

# ─── Prompt logic ────────────────────────────────────────────────────────────────
def get_prompt(crop_name):
    return f"A top-down view of a healthy {crop_name} plant at an early to middle growth stage, with visible leaves and structure"

def get_similar_classes_imagebind(candidates_classes: list[str], target_class: str):
    
    prompt = get_prompt(target_class)
    query_embedding = get_embedding(prompt)

    if not query_embedding:
        return []

    corpus_prompts = [get_prompt(crop) for crop in candidates_classes]
    corpus_embeddings = get_embeddings(corpus_prompts)
    if not corpus_embeddings:
        return []

    matches = find_matches(query_embedding, corpus_embeddings, threshold=0.2)
    if not matches:
        return []

    closest_match = matches[0]
    closest_match_score = closest_match[1]
    closest_match_index = closest_match[0]
    closest_match_class = candidates_classes[closest_match_index]

    similar_matches = []
    for idx, score in matches[1:]:
        if score >= closest_match_score - 0.08:
            similar_matches.append(candidates_classes[idx])

    return [closest_match_class] + similar_matches
