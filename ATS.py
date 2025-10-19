import numpy as np  
import re
from typing import List , Tuple  , Dict , Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


resume_text = """
Experienced Python developer with strong skills in data analysis, 
machine learning, and backend API development using Flask and FastAPI. 
Familiar with cloud tools like AWS and Docker.
"""

jd_text = """
Looking for a backend engineer skilled in Python, FastAPI, and AWS. 
Experience with Docker and machine learning is a plus.
"""





model = SentenceTransformer('all-MiniLM-L6-v2')

def ats_score(resume_text: str, jd_text: str) -> float:

    
    
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
     
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.replace('•', ' ').replace('·', ' ')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
        text = re.sub(r'[^a-zA-Z0-9\s\-\+.#/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    
    r_emb = model.encode([clean_text(resume_text)] , convert_to_numpy  = True)
    jd_emb = model.encode([clean_text(jd_text)] , convert_to_numpy = True)
    
    sim = cosine_similarity(r_emb, jd_emb)[0][0]
    return sim 


score = ats_score(resume_text, jd_text)
print(f"ATS Match Score: {score:.2f}")
