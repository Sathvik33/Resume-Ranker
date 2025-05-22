# Resume Ranker

A Streamlit web application that allows users to upload multiple PDF resumes and rank them against a job description based on semantic similarity and ATS (Applicant Tracking System) compatibility analysis. The app extracts key skills, anonymizes Personally Identifiable Information (PII), and provides insights on skill gaps and ATS readiness.

---

## Features

- Upload multiple PDF resumes and a job description.
- Extract text from PDFs with PII anonymization.
- Preprocess text using SpaCy (lemmatization, stopword removal).
- Semantic ranking of resumes using SentenceTransformers embeddings.
- ATS compatibility scoring with feedback on formatting and content.
- Skill gap analysis compared to job description requirements.
- Download ranking results as a CSV file.
- User-friendly interface powered by Streamlit.

---
