import os
import re
import pickle
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
app.secret_key = 'your_super_secret_key'
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


# [IMPROVEMENT] New function to load job requirements dynamically
def load_job_requirements_from_csv(filepath):
    """
    Loads job role requirements from a CSV file and groups skills by role.
    This makes the skill-gap analysis data-driven and easy to update.
    """
    try:
        df = pd.read_csv(filepath)
        # Standardize by stripping whitespace and converting to lowercase
        df['Job Role'] = df['Job Role'].str.strip()
        df['Required Skill'] = df['Required Skill'].str.strip().str.lower()

        # Group skills for each job role into a dictionary
        job_reqs = df.groupby('Job Role')['Required Skill'].apply(list).to_dict()

        print("Successfully loaded job requirements from CSV.")
        return job_reqs
    except FileNotFoundError:
        print(f"Warning: Job requirements file not found at {filepath}. Skill gap analysis may be incomplete.")
        return {}
    except Exception as e:
        print(f"Error loading job requirements from CSV: {e}")
        return {}


# ----------------------------------------------------
# Resume Analyzer Functions
# ----------------------------------------------------
# In app.py, replace the entire create_resume_analyzer function

def create_resume_analyzer():
    """
    Factory function to create the resume analyzer.
    This loads models and data only once for better performance.
    """
    # Load ML models and assets
    model_path = os.path.join('models', 'resume_analyzer_model.pkl')
    scaler_path = os.path.join('models', 'resume_scaler.pkl')
    features_path = os.path.join('models', 'feature_columns.pkl')

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
    except FileNotFoundError as e:
        print(f"FATAL: Could not load model files: {e}")
        print("Please ensure the training notebook has been run and files are in the 'models/' directory.")
        return None

    job_requirements_path = os.path.join('Data', 'updated_job_roles_full.csv')
    job_requirements = load_job_requirements_from_csv(job_requirements_path)

    def extract_skills_from_resume(resume_text):
        """Extracts skills from resume text."""
        resume_lower = resume_text.lower()
        skills_found = {
            'has_python': 1 if 'python' in resume_lower else 0,
            'has_java': 1 if 'java' in resume_lower and 'javascript' not in resume_lower else 0,
            'has_sql': 1 if 'sql' in resume_lower or 'mysql' in resume_lower or 'postgresql' in resume_lower else 0,
            'has_ml': 1 if 'machine learning' in resume_lower or ' ml ' in resume_lower or 'artificial intelligence' in resume_lower else 0,
            'has_cybersecurity': 1 if 'cybersecurity' in resume_lower or 'cyber security' in resume_lower or 'information security' in resume_lower else 0,
            'has_blockchain': 1 if 'blockchain' in resume_lower or 'cryptocurrency' in resume_lower else 0,
            'has_cloud': 1 if 'cloud' in resume_lower or 'aws' in resume_lower or 'azure' in resume_lower else 0,
            'has_data_analysis': 1 if 'data analysis' in resume_lower or 'data analytics' in resume_lower else 0,
            'knows_python_prog': 1 if 'python' in resume_lower else 0,
            'knows_java_prog': 1 if 'java' in resume_lower and 'javascript' not in resume_lower else 0,
            'knows_javascript': 1 if 'javascript' in resume_lower or 'js' in resume_lower else 0,
            'knows_r': 1 if ' r ' in resume_lower or 'r programming' in resume_lower else 0,
            'knows_cpp': 1 if 'c++' in resume_lower or 'cpp' in resume_lower else 0,
            'knows_csharp': 1 if 'c#' in resume_lower or 'csharp' in resume_lower else 0,

            # [FIX] Add pandas and git to the primary skill extraction
            'has_pandas': 1 if 'pandas' in resume_lower else 0,
            'has_git': 1 if 'git' in resume_lower else 0,
        }
        skills_found['technical_skills_count'] = sum(v for k, v in skills_found.items() if k.startswith('has_'))
        skills_found['programming_languages_count'] = sum(v for k, v in skills_found.items() if k.startswith('knows_'))
        skills_found.update({
            'has_projects': 1 if 'project' in resume_lower else 0,
            'technical_rating': 3,
            'soft_skills_rating': 3
        })
        return skills_found

    def generate_skill_gaps(current_skills, predicted_job):
        """Generates skill gap analysis based on predicted job."""
        required_skills = job_requirements.get(predicted_job, [])

        # This mapping helps translate extracted skills to the required skills list
        skill_mapping = {
            'python': current_skills.get('has_python', 0) or current_skills.get('knows_python_prog', 0),
            'java': current_skills.get('has_java', 0) or current_skills.get('knows_java_prog', 0),
            'sql': current_skills.get('has_sql', 0),
            'machine learning': current_skills.get('has_ml', 0),
            'javascript': current_skills.get('knows_javascript', 0),
            'cybersecurity': current_skills.get('has_cybersecurity', 0),
            'cloud computing': current_skills.get('has_cloud', 0),

            # [FIX] Use the extracted skill from the dictionary instead of re-scanning the text
            'pandas': current_skills.get('has_pandas', 0),
            'git': current_skills.get('has_git', 0),
        }
        missing_skills = [skill for skill in required_skills if not skill_mapping.get(skill, 0)]
        match_percentage = ((len(required_skills) - len(missing_skills)) / len(
            required_skills) * 100) if required_skills else 100
        return {
            'required_skills': required_skills,
            'missing_skills': missing_skills,
            'skill_match_percentage': match_percentage
        }

    def analyze_resume(resume_text, intelligence_scores=None):
        """Analyzes resume and provides job recommendations."""
        skills = extract_skills_from_resume(resume_text)
        if intelligence_scores is None:
            intelligence_scores = {
                'linguistic_score': 5, 'musical_score': 5, 'bodily_score': 5,
                'logical_mathematical_score': 5, 'spatial_visualization_score': 5,
                'interpersonal_score': 5, 'intrapersonal_score': 5, 'naturalist_score': 5
            }
        skills.update(intelligence_scores)

        # Note: 'has_pandas' and 'has_git' are not in your trained model's feature_columns.
        # This is okay, as they are only used for the skill-gap analysis, not the prediction.
        feature_vector = np.array([skills.get(col, 0) for col in feature_columns]).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)

        predicted_job = model.predict(feature_vector_scaled)[0]
        prediction_probabilities = model.predict_proba(feature_vector_scaled)[0]

        job_classes = model.classes_
        top3_indices = np.argsort(prediction_probabilities)[-3:][::-1]
        recommendations = [{'job': job_classes[i], 'probability': prediction_probabilities[i]} for i in top3_indices]

        # Now this function no longer needs the raw resume_text
        skill_gaps = generate_skill_gaps(skills, predicted_job)

        return {
            'predicted_job': predicted_job,
            'confidence': max(prediction_probabilities),
            'top_recommendations': recommendations,
            'extracted_skills': skills,
            'skill_gaps': skill_gaps
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
            analysis_result = analyze_resume(raw_text)
            session['analysis_result'] = analysis_result

            missing_skills = analysis_result['skill_gaps']['missing_skills']
            missing_skill_recos = {skill.title(): course_recos.get(skill.title(), {}) for skill in missing_skills}

            predicted_role = analysis_result['predicted_job']
            # Handle cases where model predicts 'Data Scientist' but JSON key is 'Data Science'
            career_path_key = predicted_role
            if predicted_role == 'Data Scientist':
                career_path_key = 'Data Science'

            career_path = career_recos.get(career_path_key, None)

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