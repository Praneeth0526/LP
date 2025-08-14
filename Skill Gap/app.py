import os
import re
import joblib  # Using joblib as it's more efficient for scikit-learn models
import fitz  # For PDF reading
import docx  # For DOCX reading
import pandas as pd
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_super_secret_key'  # Change this to a random string
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_file(filepath):
    """Extracts raw text from PDF or DOCX files."""
    if filepath.endswith('.pdf'):
        try:
            doc = fitz.open(filepath)
            return " ".join([page.get_text() for page in doc])
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
            return ""
    elif filepath.endswith('.docx'):
        try:
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            return ""
    return ""


def load_job_requirements_from_csv(filepath):
    """Loads job role requirements from a CSV file for skill-gap analysis."""
    try:
        df = pd.read_csv(filepath)
        # Standardize column names for consistency
        df.columns = [col.strip() for col in df.columns]
        df['Job Role'] = df['Job Role'].str.strip()
        df['Required Skill'] = df['Required Skill'].str.strip().str.lower()
        job_reqs = df.groupby('Job Role')['Required Skill'].apply(list).to_dict()
        print("Successfully loaded job requirements from CSV.")
        return job_reqs
    except Exception as e:
        print(f"Warning: Could not load job requirements from {filepath}. Error: {e}")
        return {}


# ----------------------------------------------------
# Resume Analyzer with Dual-Model Logic
# ----------------------------------------------------
def create_resume_analyzer():
    """
    Factory function to create the resume analyzer.
    This loads all models and assets only once for better performance.
    """
    try:
        skills_model = joblib.load('models/skills_model.pkl')
        holistic_model = joblib.load('models/holistic_model.pkl')
        skills_features_list = joblib.load('models/skills_features.pkl')
        holistic_features_list = joblib.load('models/holistic_features.pkl')
        career_encoder = joblib.load('models/career_label_encoder.pkl')
    except FileNotFoundError as e:
        print(f"FATAL: Could not load model files: {e}")
        return None

    all_features_list = sorted(list(set(skills_features_list + holistic_features_list)))
    job_requirements_path = os.path.join('Data', 'updated_job_roles_full.csv')
    job_requirements = load_job_requirements_from_csv(job_requirements_path)

    def extract_features_from_resume(resume_text):
        """Generates the feature set required by the models from raw resume text."""
        resume_lower = resume_text.lower()
        features = {feat: 0 for feat in all_features_list}

        for col in all_features_list:
            if col.startswith(('Tech_', 'Prog_', 'Soft_')):
                skill = col.split('_', 1)[1].replace('_', ' ').lower()
                if skill in resume_lower:
                    features[col] = 1

        features['Has_Projects_Encoded'] = 1 if 'project' in resume_lower else 0
        tech_skill_count = sum(v for k, v in features.items() if k.startswith('Tech_'))
        prog_lang_count = sum(v for k, v in features.items() if k.startswith('Prog_'))
        total_tech_skills = tech_skill_count + prog_lang_count

        features['Technical_Competency_Score'] = min(total_tech_skills / 10.0, 1.0)
        features['Skill_Diversity_Index'] = total_tech_skills
        features['Career_Readiness_Score'] = (features['Has_Projects_Encoded'] * 0.5) + (
                    features['Technical_Competency_Score'] * 0.5)

        return features

    def analyze_resume(resume_text, intelligence_scores):
        """Analyzes a resume using both the skills-based and holistic models."""
        features = extract_features_from_resume(resume_text)

        # Update features with user-provided intelligence scores
        features.update(intelligence_scores)
        features['Intelligence_Profile_Score'] = np.mean(list(intelligence_scores.values())) / 10.0

        # Predict with Skills-Based Model
        skills_vector = np.array([features.get(col, 0) for col in skills_features_list]).reshape(1, -1)
        skills_pred_index = skills_model.predict(skills_vector)[0]
        skills_pred_job = career_encoder.inverse_transform([skills_pred_index])[0]
        skills_pred_proba = skills_model.predict_proba(skills_vector)[0]

        # Predict with Holistic Model
        holistic_vector = np.array([features.get(col, 0) for col in holistic_features_list]).reshape(1, -1)
        holistic_pred_index = holistic_model.predict(holistic_vector)[0]
        holistic_pred_job = career_encoder.inverse_transform([holistic_pred_index])[0]
        holistic_pred_proba = holistic_model.predict_proba(holistic_vector)[0]

        # --- [FIX] Correct Skill Gap Analysis ---
        # Handle job title mismatches (e.g., "Data Scientist" vs "Data Science")
        job_key = skills_pred_job
        if job_key not in job_requirements and "Data" in job_key:
            job_key = "Data Science"

        required = job_requirements.get(job_key, [])

        # [FIX] Make skill checking case-insensitive and robust
        have_skills = []
        for req_skill in required:
            # Check against all feature keys, ignoring case
            for feature_key in features.keys():
                if req_skill.lower() in feature_key.lower() and features[feature_key] == 1:
                    have_skills.append(req_skill)
                    break  # Move to the next required skill once found

        missing_skills = [skill for skill in required if skill not in have_skills]
        match_percentage = (len(have_skills) / len(required) * 100) if required else 100

        skill_gaps = {
            'required_skills': required,
            'missing_skills': missing_skills,
            'skill_match_percentage': match_percentage
        }

        return {
            'skills_prediction': {'job': skills_pred_job, 'confidence': skills_pred_proba[skills_pred_index]},
            'holistic_prediction': {'job': holistic_pred_job, 'confidence': holistic_pred_proba[holistic_pred_index]},
            'skill_gaps': skill_gaps,
            'predicted_job': skills_pred_job
        }

    return analyze_resume


# -----------------------------
# Load Analyzer and Data
# -----------------------------
analyze_resume = create_resume_analyzer()
with open("Data/course_recommendations.json") as f:
    course_recos = json.load(f)
with open("Data/career_roadmap_courses.json") as f:
    career_recos = json.load(f)


# -----------------------------
# Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        if request.form['username'] == "admin" and request.form['password'] == "admin":
            session['loggedin'] = True
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)


@app.route('/home')
def home():
    if 'loggedin' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/about')
def about():
    if 'loggedin' in session:
        return render_template('about.html')
    return redirect(url_for('login'))


@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'resume_file' not in request.files or not request.files['resume_file'].filename:
            return render_template("upload.html", error="No file selected.")

        file = request.files['resume_file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        raw_text = extract_text_from_file(filepath)
        if not raw_text:
            return render_template("upload.html", error=f"Could not extract text from {filename}.")

        if analyze_resume:
            # --- [FIX] Get intelligence scores from the form ---
            intelligence_scores = {
                'Linguistic': int(request.form.get('Linguistic', 5)),
                'Musical': int(request.form.get('Musical', 5)),
                'Bodily': int(request.form.get('Bodily', 5)),
                'Logical - Mathematical': int(request.form.get('Logical - Mathematical', 5)),
                'Spatial-Visualization': int(request.form.get('Spatial-Visualization', 5)),
                'Interpersonal': int(request.form.get('Interpersonal', 5)),
                'Intrapersonal': int(request.form.get('Intrapersonal', 5)),
                'Naturalist': int(request.form.get('Naturalist', 5))
            }

            analysis_result = analyze_resume(raw_text, intelligence_scores)
            session['analysis_result'] = analysis_result

            missing_skills = analysis_result['skill_gaps']['missing_skills']
            missing_skill_recos = {skill.title(): course_recos.get(skill.title(), {}) for skill in missing_skills}

            predicted_role = analysis_result['predicted_job']
            career_path = career_recos.get(predicted_role, None)

            # Handle job title mismatches for career path as well
            if not career_path and 'Data' in predicted_role:
                career_path = career_recos.get('Data Science', None)

            return render_template("result.html",
                                   result=analysis_result,
                                   recommendations=missing_skill_recos,
                                   career=career_path)
        else:
            return render_template("upload.html", error="Resume analyzer is not available. Check server logs.")

    return render_template("upload.html")


@app.route('/detailed_analysis')
def detailed_analysis():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    result = session.get('analysis_result', None)
    if not result:
        return redirect(url_for('upload_resume'))
    return render_template('detailed_analysis.html', result=result)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    return redirect(url_for('login'))


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)