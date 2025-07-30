import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ─── Load Llama Text Encoder ─────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-hf" # Placeholder, actual Llama model might be different

tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModel.from_pretrained(model_name).to(device)
llama_model.eval()

# ─── Embedding helpers ────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Compute a single text embedding via Llama, normalized to unit length."""
    if not text:
        return []
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = llama_model(**inputs)
        # Using the mean of the last hidden state as the embedding
        text_emb = outputs.last_hidden_state.mean(dim=1)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    return text_emb.cpu().numpy().flatten().tolist()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute Llama embeddings for a list of texts."""
    if not texts:
        return []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = llama_model(**inputs)
        text_embs = outputs.last_hidden_state.mean(dim=1)
    text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
    return text_embs.cpu().numpy().tolist()

# ─── Similarity search ────────────────────────────────────────────────────────────
def find_matches(query_embedding: list[float],
                 corpus_embeddings: list[list[float]],
                 ):
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

def get_similar_classes_llama(candidates_classes: list[str],
                    target_class: str, threshold=0.08):
    prompt = get_prompt(target_class)
    query_embedding = get_embedding(prompt)

    if not query_embedding:
        return []

    corpus_prompts = [get_prompt(crop) for crop in candidates_classes]
    corpus_embeddings = get_embeddings(corpus_prompts)
    if not corpus_embeddings:
        return []

    matches = find_matches(query_embedding, corpus_embeddings)
    if not matches:
        return []

    closest_match = matches[0]
    closest_match_score = closest_match[1]
    closest_match_index = closest_match[0]
    closest_match_class = candidates_classes[closest_match_index]

    similar_matches = []
    for idx, score in matches[1:]:
        if score >= closest_match_score - threshold:
            similar_matches.append(candidates_classes[idx])

    return [closest_match_class] + similar_matches