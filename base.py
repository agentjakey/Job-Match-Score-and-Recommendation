import os
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer, util
from pyresparser import ResumeParser

# -----------------------------
# Load Kaggle Dataset
# -----------------------------

print('Downloading dataset...')
path = kagglehub.dataset_download("asaniczka/1-3m-linkedin-jobs-and-skills-2024")
skills_df = pd.read_csv(ps.path.join(path, 'job_skills.csv')

# -----------------------------
# Load Embedding Model
# -----------------------------

model = SentenceTransformer('all-MiniLM-L6-v2') #fast, lightweight, test dif. methods later
                        
# -----------------------------
# Parse Resume (test manual parse later for faster, more specific)
# -----------------------------

def parse_resume(resume_path):
  parsed = ResumerParser(resume_path).get_extracted_data()
  resume_text = ' '.join([str(v) for v in parsed.values() if v])
  return resume_text, parsed.get('skills', [])

# -----------------------------
# Compute Match Score
# -----------------------------

def compute_match_score(resume_text, job_text):
  resume_emb = model.encode(resume_text, convert_to_tensor = True)
  job_emb = model.encode(job_text, convert_to_tensor = True)
  score = util.pytorch_cos_sim(resume_emb, job_emb).item()
  return round(score * 100, 2) 

# -----------------------------
# Generate Improvement Suggestions
# -----------------------------

def suggest_improvements(resume_skills, job_text):
  job_skills = [s.lower() for s in skills_df['skill'].dopna().unique() if s.lower() in job_text.lower()]
  missing_skills = [s for s in job_skills if s.lower() not in [r.lower() for r in resume_skills]]

suggestions = []
if missing_skills:
  suggestions.append(f'Add or emphasize these relevant skills: {', '.join(missing_skills)}')
if len(resume_skills) < len(job_skills):
  suggestions.append('Highlight more experiences that match the job requirements.)
if not suggestions:
  sugguestions.append('Your resume is already highly aligned with the job posting. Great job!)
return suggestions

# -----------------------------
# Exec.
# -----------------------------

resume_file = 'resume.pdf' #change to actual 
job_description = 'We are seeking a Machine Learning Engineer with experience in Python, TensorFlow, and NLP.
Responsibilities include building and deploying AI models, working with cloud infrastructure,
and collaborating with data scientists. Preferred skills include Kubernetes, PyTorch, and SQL.' #change to actual

resume_text, resume_skills = parse_resume(resume_file)

match_score = compute_match_score(resume_text, job_description)
suggestions = suggest_improvements(resume_skills, job_description)

print(f"Match Score: {match_score}%")
print("\nRecommendations:")
for s in suggestions:
    print(f"- {s}")
