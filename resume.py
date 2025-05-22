import os
import streamlit as st
import pandas as pd
import spacy
import tempfile
from utils.extract_text import extract_text_from_pdf
from utils.semantic_ranker import rank_semantic
from presidio_analyzer import AnalyzerEngine
from spacy.language import Language
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load SpaCy model and Presidio for PII anonymization
nlp = spacy.load("en_core_web_sm")
analyzer = AnalyzerEngine()

# Custom SpaCy component to identify sections
@Language.component("section_identifier")
def section_identifier(doc: Doc) -> Doc:
    return doc

# Initialize pipeline
if "section_identifier" not in nlp.pipe_names:
    nlp.add_pipe("section_identifier")

if "entity_ruler" not in nlp.pipe_names:
    ruler = EntityRuler(nlp, overwrite_ents=True)
    patterns = [
        {"label": "SECTION", "pattern": [{"LOWER": {"IN": ["skills", "experience", "education"]}}]}
    ]
    ruler.add_patterns(patterns)
    nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

def preprocess_text(text: str) -> str:
    """Preprocess text: lowercase, lemmatize, keep nouns/verbs, remove PII."""
    try:
        results = analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
        for result in results:
            text = text[:result.start] + "*" * (result.end - result.start) + text[result.end:]

        text = text.lower()
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop]
        processed_text = " ".join(tokens)
        logger.info(f"Preprocessed text length: {len(processed_text.split())} tokens")
        return processed_text
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        st.error(f"Text preprocessing failed: {e}")
        return ""

def extract_skills(text: str) -> set:
    """Extract skills from text for gap analysis."""
    doc = nlp(text)
    skills = set()
    common_skills = {
        "python", "java", "javascript", "sql", "django", "flask", "machine learning", "data analysis",
        "aws", "docker", "kubernetes", "tensorflow", "pytorch", "react", "angular", "node.js"
    }

    for ent in doc.ents:
        if ent.label_ == "SECTION" and ent.text.lower() == "skills":
            start = ent.end
            for token in doc[start:]:
                if token.ent_type_ == "SECTION":
                    break
                if token.is_alpha and not token.is_stop:
                    skills.add(token.lemma_)

    for token in doc:
        if token.lemma_ in common_skills:
            skills.add(token.lemma_)

    logger.info(f"Extracted skills: {skills}")
    return skills

def check_ats_compatibility(text: str) -> int:
    """Check ATS compatibility of a resume, return score only."""
    ats_score = 100
    
    if not text.strip():
        ats_score -= 50
    if len(text.split()) < 50:
        ats_score -= 20
    if any(keyword in text.lower() for keyword in ["graphic", "image", "table"]):
        ats_score -= 30

    return max(0, ats_score)

st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("Resume Ranker")
st.write("Upload PDF resumes and enter a job description to rank them using semantic similarity and analyze ATS compatibility.")

# User inputs
job_desc = st.text_area("Enter Job Description", height=150)
uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_desc:
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        progress_bar = st.progress(0)
        job_desc_processed = preprocess_text(job_desc)
        if not job_desc_processed:
            st.error("Job description could not be processed properly.")
            st.stop()

        job_skills = extract_skills(job_desc)
        resume_texts = {}
        ats_results = {}
        skill_gaps = {}
        skipped_files = 0
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                text = extract_text_from_pdf(tmp_path)
                os.remove(tmp_path)

                if not text.strip():
                    st.warning(f"{uploaded_file.name} contains no extractable text.")
                    skipped_files += 1
                    continue

                ats_score = check_ats_compatibility(text)
                ats_results[uploaded_file.name] = ats_score

                processed = preprocess_text(text)
                if not processed or len(processed.split()) < 10:
                    st.warning(f"Processed text for {uploaded_file.name} is too short or empty. Skipping.")
                    skipped_files += 1
                    continue

                resume_texts[uploaded_file.name] = processed
                resume_skills = extract_skills(text)
                skill_gaps[uploaded_file.name] = job_skills - resume_skills

            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {e}")
                skipped_files += 1
            finally:
                progress_bar.progress((i + 1) / total_files)

        valid_count = len(resume_texts)
        st.write(f"Processed {valid_count} valid resumes out of {total_files} uploaded.")
        if skipped_files > 0:
            st.warning(f"Skipped {skipped_files} files due to processing issues.")

        if not resume_texts:
            st.error("No valid resumes were processed. Please upload readable PDF resumes.")
        else:
            try:
                ranked = rank_semantic(job_desc_processed, resume_texts)
                logger.info(f"Ranked output to DataFrame: {ranked}")

                # Validate ranked output
                if not ranked:
                    raise ValueError("No ranking results returned.")
                if not all(len(row) == 2 for row in ranked):
                    logger.error(f"Expected 2 columns (Resume, Score), got: {ranked}")
                    raise ValueError(f"Expected 2 columns (Resume, Score), got {len(ranked[0])} columns")

                df_results = pd.DataFrame(ranked, columns=["Resume", "Score"])
                df_results["ATS Score"] = df_results["Resume"].map(lambda x: ats_results.get(x, 0))
                df_results["Skill Gaps"] = df_results["Resume"].map(lambda x: ", ".join(skill_gaps.get(x, set())) or "None")

                # Display ATS score summary
                st.subheader("ATS Compatibility Summary")
                for resume in df_results["Resume"]:
                    ats_score = df_results[df_results["Resume"] == resume]["ATS Score"].iloc[0]
                    st.write(f"{resume}: ATS Score = {ats_score:.0f}")

                st.success("Ranking complete!")
                st.dataframe(df_results.style.format({"Score": "{:.2f}", "ATS Score": "{:.0f}"}))

                csv_path = os.path.join(tempfile.gettempdir(), "ranked_results.csv")
                df_results.to_csv(csv_path, index=False)
                with open(csv_path, "rb") as f:
                    st.download_button("Download Results as CSV", f, file_name="ranked_resumes.csv")

            except Exception as e:
                logger.error(f"Error ranking resumes: {e}")
                st.error(f"Error ranking resumes: {e}")