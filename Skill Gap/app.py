import os
import joblib
import fitz
import docx
import pandas as pd
import numpy as np
import json
import re
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_super_secret_key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Load All ML Artifacts and Data
# -----------------------------
try:
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')

    job_requirements_df = pd.read_csv('Data/csv_sheets/Job_Roles_And_Skills.csv')
    job_requirements_df.columns = [col.strip() for col in job_requirements_df.columns]
    job_requirements = job_requirements_df.groupby('Job Role')['Required Skill'].apply(list).to_dict()

    with open("Data/course_recommendations.json") as f:
        course_recos = json.load(f)
    with open("Data/career_roadmap_courses.json") as f:
        career_recos = json.load(f)

    print("✅ All models and data files loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ FATAL ERROR: Could not load model files: {e}")
    print("Please ensure you have run the final training notebook to generate all .pkl files.")
    model = None


# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_file(filepath):
    """Extracts raw text from PDF or DOCX files."""
    if filepath.endswith('.pdf'):
        try:
            return " ".join([page.get_text() for page in fitz.open(filepath)])
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
            return ""
    elif filepath.endswith('.docx'):
        try:
            return "\n".join([para.text for para in docx.Document(filepath).paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {filepath}: {e}")
            return ""
    return ""


def extract_resume_highlights(text):
    """
    Extracts key sections like Education, Experience, and Skills from resume text.
    """
    highlights = {}
    # Define patterns for common resume sections. Case-insensitive.
    patterns = {
        'Education': r'(?i)\b(education|academic background|qualifications)\b',
        'Experience': r'(?i)\b(experience|work experience|professional experience|employment history)\b',
        'Skills': r'(?i)\b(skills|technical skills|core competencies)\b',
        'Projects': r'(?i)\b(projects|personal projects|academic projects)\b'
    }

    # Find all section headers and their positions
    found_sections = []
    for section, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            found_sections.append((section, match.start()))

    # Sort sections by their appearance in the text
    found_sections.sort(key=lambda x: x[1])

    if not found_sections:
        # If no sections found, return a snippet of the text
        return {'Summary': text[:700] + '...'}

    # Extract text between sections
    for i, (section, start_pos) in enumerate(found_sections):
        # Determine the end position of the current section's text
        end_pos = found_sections[i+1][1] if i + 1 < len(found_sections) else len(text)

        # Extract the text block for the section
        section_text = text[start_pos:end_pos].strip()

        # Clean up the section text: remove the header itself and limit length
        header_match = re.search(patterns[section], section_text)
        if header_match:
            # Start text after the header
            text_start = header_match.end()
            # Clean up leading characters like newlines or colons
            content = re.sub(r'^[\s:.-]*', '', section_text[text_start:])
            # Limit to a reasonable length for display
            highlights[section] = content[:700] + '...' if len(content) > 700 else content

    return highlights


# -----------------------------
# Core Prediction Pipeline
# -----------------------------
def create_feature_vector(resume_text, form_data):
    """
    Creates a feature vector from raw inputs that matches the model's training data.
    """
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0
    resume_lower = resume_text.lower()

    # 1. Extract features from resume text (Year and Course)
    # Extract Academic Year
    year_map = {'1st': 1, 'first': 1, '2nd': 2, 'second': 2, '3rd': 3, 'third': 3, '4th': 4, 'final': 4, 'fourth': 4}
    year_found = 0
    year_match = re.search(r'\b(1st|2nd|3rd|4th|first|second|third|fourth|final)[\s-]?year\b', resume_lower)
    if year_match:
        year_key = year_match.group(1)
        year_found = year_map.get(year_key, 0)

    if 'Year' in input_df.columns:
        input_df.loc[0, 'Year'] = year_found if year_found != 0 else 2  # Default to 2nd year

    # Extract and Encode Current Course
    if 'Current Course' in label_encoders:
        course_encoder = label_encoders['Current Course']
        found_course = None
        for course in course_encoder.classes_:
            if re.search(r'\b' + re.escape(course.lower()) + r'\b', resume_lower):
                found_course = course
                break

        value_to_encode = found_course if found_course else 'B.Tech CSE'
        if value_to_encode in course_encoder.classes_:
            encoded_value = course_encoder.transform([value_to_encode])[0]
            input_df.loc[0, 'Current Course_encoded'] = encoded_value
        else:
            input_df.loc[0, 'Current Course_encoded'] = 0

    # 2. Process Numerical Features from the form
    numerical_features = [col for col in feature_columns if col in scaler.feature_names_in_]
    for col in numerical_features:
        if col in form_data:
            input_df.loc[0, col] = float(form_data.get(col, 0))

    # Create proxy for 'Score' and 'Rating' based on intelligence scores
    intel_scores = [float(v) for k, v in form_data.items() if
                    k in ['Student-Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 'Spatial-Visualization',
                          'Interpersonal', 'Intrapersonal', 'Naturalist']]
    avg_intel_score = np.mean(intel_scores) if intel_scores else 10
    input_df.loc[0, 'Score'] = avg_intel_score * 10
    input_df.loc[0, 'Rating(technical and programming)'] = float(form_data.get('Logical - Mathematical', 10))
    input_df.loc[0, 'Rating(soft skills)'] = float(form_data.get('Interpersonal', 10))

    # 3. Process other Label-Encoded Categorical Features from the form
    for col, encoder in label_encoders.items():
        if col == 'Current Course': continue  # Already handled
        encoded_col_name = f'{col}_encoded'
        if encoded_col_name in input_df.columns:
            value = form_data.get(col, encoder.classes_[0])
            if value in encoder.classes_:
                input_df.loc[0, encoded_col_name] = encoder.transform([value])[0]
            else:
                input_df.loc[0, encoded_col_name] = 0

    # 4. Process One-Hot Encoded Skill Features from the resume text
    for col in feature_columns:
        if col.startswith(('Tech_', 'Prog_', 'Soft_')):
            skill = col.split('_', 1)[1].replace('_', ' ').lower()
            if skill in resume_lower:
                input_df.loc[0, col] = 1

    # 5. Scale the numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df


def analyze_resume_and_predict(resume_text, form_data, desired_job):
    """
    Full pipeline to process inputs, make predictions, and perform skill gap analysis.
    """
    feature_vector = create_feature_vector(resume_text, form_data)

    # 1. Make Predictions
    probabilities = model.predict_proba(feature_vector)[0]

    # Get the top prediction
    top_prediction_index = np.argmax(probabilities)
    predicted_job = target_encoder.inverse_transform([top_prediction_index])[0]
    confidence = probabilities[top_prediction_index]

    # Get all high-confidence jobs (80%+)
    high_confidence_jobs = []
    for i, prob in enumerate(probabilities):
        if prob >= 0.80:
            job_name = target_encoder.inverse_transform([i])[0]
            if job_name != predicted_job:  # Don't list the top one again
                high_confidence_jobs.append({'role': job_name, 'confidence': prob})

    # 2. Perform Skill-Gap Analysis against the USER'S DESIRED JOB
    job_key = desired_job
    if job_key not in job_requirements and "Data" in job_key:
        job_key = "Data Science"

    required_skills = [skill.lower() for skill in job_requirements.get(job_key, [])]

    have_skills = []
    resume_lower = resume_text.lower()
    for skill in required_skills:
        if skill in resume_lower:
            have_skills.append(skill)

    missing_skills = [skill for skill in required_skills if skill not in have_skills]
    match_percentage = (len(have_skills) / len(required_skills) * 100) if required_skills else 100

    skill_gaps = {
        'required_skills': required_skills,
        'have_skills': have_skills,
        'missing_skills': missing_skills,
        'skill_match_percentage': match_percentage
    }

    return {
        'predicted_job': predicted_job,
        'confidence': confidence,
        'desired_job': desired_job,
        'high_confidence_jobs': sorted(high_confidence_jobs, key=lambda x: x['confidence'], reverse=True),
        'skill_gaps': skill_gaps
    }


# -----------------------------
# Flask Routes
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

        if model:
            desired_job = request.form.get('desired_job')

            # Create a mutable copy of the form data to pass to the model
            form_data_for_model = request.form.to_dict()
            # Use the user's desired job as their career interest for prediction
            form_data_for_model['Career Interest'] = desired_job

            analysis_result = analyze_resume_and_predict(raw_text, form_data_for_model, desired_job)

            # Extract highlights from the resume text
            resume_highlights = extract_resume_highlights(raw_text)

            missing_skills = analysis_result['skill_gaps']['missing_skills']
            missing_skill_recos = {skill.title(): course_recos.get(skill.title(), {}) for skill in missing_skills}

            # Use the desired job for the career roadmap
            career_path_role = analysis_result['desired_job']
            career_path = career_recos.get(career_path_role, None)
            if not career_path and 'Data' in career_path_role:
                career_path = career_recos.get('Data Science', None)

            return render_template("result.html",
                                   result=analysis_result,
                                   recommendations=missing_skill_recos,
                                   career=career_path,
                                   resume_highlights=resume_highlights)
        else:
            return render_template("upload.html", error="Model is not available. Check server logs.")

    # Pass options for dropdowns to the template
    form_options = {
        'Projects': label_encoders['Projects'].classes_,
        'Career Interest': label_encoders['Career Interest'].classes_,
        'Challenges': label_encoders['Challenges'].classes_,
        'Support-required': label_encoders['Support-required'].classes_,
        'All_Jobs': sorted(list(job_requirements.keys()))
    }
    return render_template("upload.html", form_options=form_options)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    return redirect(url_for('login'))


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)