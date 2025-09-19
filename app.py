from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

# Request structure
class TicketText(BaseModel):
    key: str
    text: str

class SimilarityRequest(BaseModel):
    current: TicketText
    others: List[TicketText]

# Response structure
class SimilarityScore(BaseModel):
    key: str
    score: float

@app.post("/similarity", response_model=List[SimilarityScore])
async def calculate_similarity(payload: SimilarityRequest):
    print("Inside similarity api:")
    current_embedding = model.encode(payload.current.text, convert_to_tensor=True)
    others_text = [t.text for t in payload.others]
    others_keys = [t.key for t in payload.others]

    if not others_text:
        return []

    other_embeddings = model.encode(others_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(current_embedding, other_embeddings)[0]

    results = [
        {"key": key, "score": round(score.item(), 4)}
        for key, score in zip(others_keys, cosine_scores)
    ]
    return results
