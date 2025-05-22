from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from joblib import Memory
import spacy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for TF-IDF vectors
memory = Memory("cache", verbose=0)
nlp = spacy.load("en_core_web_sm")

def extract_key_criteria(text: str) -> dict:
    """Extract skills and experience years from text."""
    doc = nlp(text)
    skills = set(token.lemma_ for token in doc if token.pos_ == "NOUN" and not token.is_stop)
    experience_years = 0
    for ent in doc.ents:
        if ent.label_ in ["DATE", "QUANTITY"] and any(w in ent.text.lower() for w in ["year", "years"]):
            try:
                experience_years = max(experience_years, int(ent.text.split()[0]))
            except (ValueError, IndexError):
                pass
    return {"skills": skills, "experience_years": experience_years}

@memory.cache
def compute_tfidf_vectors(documents: list) -> tuple:
    """Compute TF-IDF vectors with caching."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)
    return vectorizer, vectors

def rank_tfidf(job_description: str, resume_texts: dict) -> list:
    try:
        names = list(resume_texts.keys())
        documents = list(resume_texts.values())
        all_docs = [job_description] + documents

        # Compute TF-IDF vectors
        vectorizer, vectors = compute_tfidf_vectors(all_docs)
        job_vec = vectors[0]
        resume_vecs = vectors[1:]

        # Base cosine similarity scores
        tfidf_scores = cosine_similarity(job_vec, resume_vecs).flatten()

        # Weighted scoring for skills and experience
        job_criteria = extract_key_criteria(job_description)
        weights = {"skills": 0.6, "experience": 0.4}  # Adjustable weights
        final_scores = []

        for i, name in enumerate(names):
            resume_criteria = extract_key_criteria(resume_texts[name])
            skill_overlap = len(job_criteria["skills"].intersection(resume_criteria["skills"])) / max(1, len(job_criteria["skills"]))
            experience_score = min(resume_criteria["experience_years"] / max(1, job_criteria["experience_years"]), 1.0)
            weighted_score = (weights["skills"] * skill_overlap + weights["experience"] * experience_score) * 0.3 + tfidf_scores[i] * 0.7
            final_scores.append(weighted_score)

        ranking = sorted(zip(names, final_scores), key=lambda x: x[1], reverse=True)
        ranked_df = pd.DataFrame(ranking, columns=["Resume", "Score"])
        return ranked_df.values.tolist()

    except Exception as e:
        logger.error(f"TF-IDF ranking failed: {e}")
        raise