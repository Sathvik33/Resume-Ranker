from sentence_transformers import SentenceTransformer, util
import pandas as pd
from joblib import Memory
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for embeddings
memory = Memory("cache", verbose=0)
model = SentenceTransformer('all-MiniLM-L6-v2')

@memory.cache
def compute_embeddings(texts: list) -> list:
    """Compute sentence transformer embeddings with caching."""
    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        logger.info(f"Computed embeddings for {len(texts)} texts")
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Embedding computation failed: {e}")
        raise

def rank_semantic(job_description: str, resume_texts: dict) -> list:
    try:
        names = list(resume_texts.keys())
        documents = list(resume_texts.values())
        logger.info(f"Processing {len(names)} resumes for semantic ranking")

        # Compute embeddings
        all_texts = [job_description] + documents
        embeddings = compute_embeddings(all_texts)
        job_embedding = embeddings[0]
        resume_embeddings = embeddings[1:]

        # Calculate cosine similarity
        scores_tensor = util.cos_sim(job_embedding, resume_embeddings)
        logger.info(f"Cosine similarity tensor shape: {scores_tensor.shape}")

        # Handle single vs. multiple resumes
        if len(resume_texts) == 1:
            scores = [float(scores_tensor.item())]
        else:
            scores = scores_tensor.squeeze().tolist()
            if isinstance(scores, float):
                scores = [scores]
        
        # Normalize scores to 60-100
        normalized_scores = [max(60, score * 100) for score in scores]
        logger.info(f"Normalized semantic scores: {normalized_scores}")

        # Create ranking with only Resume and Score
        ranking = [[name, score] for name, score in sorted(zip(names, normalized_scores), key=lambda x: x[1], reverse=True)]
        logger.info(f"Ranking output: {ranking}")
        return ranking

    except Exception as e:
        logger.error(f"Semantic ranking failed: {e}")
        raise