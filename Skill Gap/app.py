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


# -----------------------------
# Core Prediction Pipeline
# -----------------------------
def create_feature_vector(resume_text, form_data):
    """
    Creates a feature vector from raw inputs that matches the model's training data.
    """
    # 1. Initialize a DataFrame with all the required feature columns, filled with zeros
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # 2. Process Numerical Features from the form
    numerical_features = [col for col in feature_columns if col in scaler.feature_names_in_]
    for col in numerical_features:
        if col in form_data:
            input_df.loc[0, col] = float(form_data.get(col, 0))

    # Create proxy for 'Score' and 'Rating' based on intelligence scores
    # [FIX] Convert the string values from the form to float before calculating the mean.
    intel_scores = [float(v) for k, v in form_data.items() if
                    k in ['Student-Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 'Spatial-Visualization',
                          'Interpersonal', 'Intrapersonal', 'Naturalist']]
    avg_intel_score = np.mean(intel_scores) if intel_scores else 5
    input_df.loc[0, 'Score'] = avg_intel_score * 10
    input_df.loc[0, 'Rating(technical and programming)'] = float(form_data.get('Logical - Mathematical', 5))
    input_df.loc[0, 'Rating(soft skills)'] = float(form_data.get('Interpersonal', 5))

    # 3. Process Label-Encoded Categorical Features from the form
    for col, encoder in label_encoders.items():
        encoded_col_name = f'{col}_encoded'
        if encoded_col_name in input_df.columns:
            value = form_data.get(col, encoder.classes_[0])
            if value in encoder.classes_:
                input_df.loc[0, encoded_col_name] = encoder.transform([value])[0]
            else:
                input_df.loc[0, encoded_col_name] = 0  # Default for unknown

    # 4. Process One-Hot Encoded Skill Features from the resume text
    resume_lower = resume_text.lower()
    for col in feature_columns:
        if col.startswith(('Tech_', 'Prog_', 'Soft_')):
            # Recreate the original skill name from the feature name
            skill = col.split('_', 1)[1].replace('_', ' ').lower()
            if skill in resume_lower:
                input_df.loc[0, col] = 1

    # 5. Scale the numerical features using the loaded scaler
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    return input_df


def analyze_resume_and_predict(resume_text, form_data):
    """
    Full pipeline to process inputs, make a prediction, and perform skill gap analysis.
    """
    # 1. Create the feature vector for the model
    feature_vector = create_feature_vector(resume_text, form_data)

    # 2. Make a prediction
    prediction_index = model.predict(feature_vector)[0]
    prediction_proba = model.predict_proba(feature_vector)[0]
    predicted_job = target_encoder.inverse_transform([prediction_index])[0]
    confidence = prediction_proba[prediction_index]

    # 3. Perform Skill-Gap Analysis
    job_key = predicted_job
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
        'skill_gaps': skill_gaps
    }


# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    # Your existing login logic
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
            analysis_result = analyze_resume_and_predict(raw_text, request.form)

            missing_skills = analysis_result['skill_gaps']['missing_skills']
            missing_skill_recos = {skill.title(): course_recos.get(skill.title(), {}) for skill in missing_skills}

            predicted_role = analysis_result['predicted_job']
            career_path = career_recos.get(predicted_role, None)

            if not career_path and 'Data' in predicted_role:
                career_path = career_recos.get('Data Science', None)

            return render_template("result.html",
                                   result=analysis_result,
                                   recommendations=missing_skill_recos,
                                   career=career_path)
        else:
            return render_template("upload.html", error="Model is not available. Check server logs.")

    # Pass unique values for dropdowns to the template
    form_options = {
        'Current Course': label_encoders['Current Course'].classes_,
        'Projects': label_encoders['Projects'].classes_,
        'Career Interest': label_encoders['Career Interest'].classes_,
        'Challenges': label_encoders['Challenges'].classes_,
        'Support-required': label_encoders['Support-required'].classes_
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