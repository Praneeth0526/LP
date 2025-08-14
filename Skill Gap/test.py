import os
import re
import pickle
import fitz  # For PDF reading
import docx  # For DOCX reading
import pandas as pd
import numpy as np
import json
from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from datetime import datetime

# -----------------------------
# Flask Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add this for session management
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Models and Data
# -----------------------------
model = pickle.load(open("models/resume_classifier_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
job_df = pd.read_csv("Data/updated_job_roles_full.csv")

# Load integrated data from your Excel file
dma_data = pd.read_excel("Data/combined_report.xlsx", sheet_name='DMA')
original_data = pd.read_excel("Data/combined_report.xlsx", sheet_name='original')
mapping_data = pd.read_excel("Data/combined_report.xlsx", sheet_name='Mapping')
job_skills_data = pd.read_excel("Data/combined_report.xlsx", sheet_name='Job_Roles_And_Skills')

# Load course and career recommendations
with open("Data/course_recommendations.json") as f:
    course_recos = json.load(f)

with open("Data/career_roadmap_courses.json") as f:
    career_recos = json.load(f)


# -----------------------------
# Skill Gap Analysis Functions
# -----------------------------

def standardize_skills(skill_string):
    """Standardize skill names and handle variations"""
    if pd.isna(skill_string) or skill_string == '':
        return []

    # Common skill standardization mappings
    skill_mappings = {
        'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
        'python': ['python', 'python programming'],
        'java': ['java', 'core java', 'java programming'],
        'sql': ['sql', 'mysql', 'database', 'rdbms'],
        'data analysis': ['data analysis', 'data analytics', 'analytics'],
        'cybersecurity': ['cybersecurity', 'cyber security', 'information security'],
        'cloud computing': ['cloud computing', 'cloud', 'aws', 'azure'],
        'javascript': ['javascript', 'js', 'node.js'],
        'blockchain': ['blockchain', 'cryptocurrency', 'smart contracts']
    }

    skills = [skill.strip().lower() for skill in skill_string.split(',')]
    standardized_skills = []

    for skill in skills:
        for standard_skill, variations in skill_mappings.items():
            if skill in variations:
                standardized_skills.append(standard_skill)
                break
        else:
            standardized_skills.append(skill)

    return list(set(standardized_skills))  # Remove duplicates


def categorize_gap_severity(match_percentage):
    """Categorize skill gap severity"""
    if match_percentage >= 80:
        return {"level": "Minimal Gap", "color": "success", "action": "Minor upskilling needed"}
    elif match_percentage >= 60:
        return {"level": "Moderate Gap", "color": "warning", "action": "Focused skill development required"}
    elif match_percentage >= 40:
        return {"level": "Significant Gap", "color": "danger", "action": "Intensive training recommended"}
    else:
        return {"level": "Major Gap", "color": "danger", "action": "Career transition support needed"}


def prioritize_missing_skills(missing_skills, predicted_role):
    """Prioritize missing skills based on importance"""
    # Define skill importance weights
    skill_importance = {
        'python': 10, 'java': 9, 'sql': 10, 'machine learning': 9,
        'javascript': 8, 'data analysis': 9, 'cybersecurity': 8,
        'cloud computing': 8, 'blockchain': 7, 'communication': 8,
        'leadership': 7, 'project management': 8, 'problem solving': 9
    }

    # Role-specific skill priorities
    role_priorities = {
        'Data Scientist': ['python', 'sql', 'machine learning', 'data analysis'],
        'Software Developer': ['java', 'python', 'javascript', 'sql'],
        'Cybersecurity Expert': ['cybersecurity', 'network security', 'python'],
        'Cloud Architect': ['cloud computing', 'aws', 'azure', 'devops'],
        'Business Analyst': ['sql', 'data analysis', 'excel', 'communication']
    }

    # Get role-specific priorities
    role_priority_skills = role_priorities.get(predicted_role, [])

    # Sort missing skills by importance
    prioritized = sorted(missing_skills,
                         key=lambda x: (
                             1 if x.lower() in role_priority_skills else 0,
                             skill_importance.get(x.lower(), 5)
                         ),
                         reverse=True)
    return prioritized


def generate_learning_path(missing_skills, predicted_role):
    """Generate structured learning path"""
    learning_phases = {
        'immediate': [],
        'short_term': [],
        'long_term': []
    }

    for i, skill in enumerate(missing_skills[:12]):  # Top 12 missing skills
        if i < 4:
            learning_phases['immediate'].append({
                'skill': skill,
                'timeline': '1-2 weeks',
                'resources': course_recos.get(skill, {"courses": ["Online tutorials", "YouTube videos"]}),
                'priority': 'High'
            })
        elif i < 8:
            learning_phases['short_term'].append({
                'skill': skill,
                'timeline': '1-3 months',
                'resources': course_recos.get(skill, {"courses": ["Online courses", "Certification programs"]}),
                'priority': 'Medium'
            })
        else:
            learning_phases['long_term'].append({
                'skill': skill,
                'timeline': '3-6 months',
                'resources': course_recos.get(skill, {"courses": ["Degree programs", "Advanced certifications"]}),
                'priority': 'Low'
            })

    return learning_phases


def analyze_holistic_skills(intelligence_scores, predicted_role):
    """Analyze holistic skills based on intelligence assessment"""
    if not intelligence_scores:
        return {
            'intelligence_profile': {},
            'role_fit_score': 75,
            'alternative_careers': [],
            'development_areas': []
        }

    # Calculate role fit based on intelligence mapping from original data
    role_requirements = {
        'Data Scientist': {'logical_mathematical': 8, 'intrapersonal': 7, 'naturalist': 6},
        'Software Developer': {'logical_mathematical': 9, 'intrapersonal': 8, 'linguistic': 6},
        'Cybersecurity Expert': {'logical_mathematical': 8, 'intrapersonal': 9, 'bodily': 6},
        'Cloud Architect': {'logical_mathematical': 9, 'spatial': 7, 'intrapersonal': 7},
        'Business Analyst': {'logical_mathematical': 8, 'interpersonal': 8, 'linguistic': 7}
    }

    requirements = role_requirements.get(predicted_role, {})
    fit_score = 75  # Default

    if requirements:
        total_fit = 0
        for intelligence, required_score in requirements.items():
            user_score = intelligence_scores.get(intelligence, 5)
            total_fit += min(user_score / required_score * 100, 100)
        fit_score = total_fit / len(requirements)

    # Suggest alternative careers if fit is low
    alternative_careers = []
    if fit_score < 70:
        for role, reqs in role_requirements.items():
            if role != predicted_role:
                role_fit = sum(min(intelligence_scores.get(intel, 5) / score * 100, 100)
                               for intel, score in reqs.items()) / len(reqs)
                if role_fit > fit_score:
                    alternative_careers.append({'role': role, 'fit_score': round(role_fit, 2)})

    # Identify development areas
    development_areas = []
    for intelligence, score in intelligence_scores.items():
        if score < 6:
            development_areas.append(intelligence)

    return {
        'intelligence_profile': intelligence_scores,
        'role_fit_score': round(fit_score, 2),
        'alternative_careers': sorted(alternative_careers, key=lambda x: x['fit_score'], reverse=True)[:3],
        'development_areas': development_areas
    }


def comprehensive_skill_gap_analysis(predicted_role, found_skills, assessment_data=None):
    """Comprehensive skill gap analysis including holistic assessment"""
    # Technical gap analysis
    required_skills_from_job_df = job_df[job_df["Job Role"] == predicted_role]["Required Skill"].tolist()

    # Get additional required skills from job_skills_data
    additional_required = job_skills_data[job_skills_data["Job Role"] == predicted_role]["Required Skill"].tolist()

    # Combine all required skills
    all_required_skills = list(set(required_skills_from_job_df + additional_required))

    # Standardize found skills
    standardized_found = []
    for skill in found_skills:
        standardized_found.extend(standardize_skills(skill))

    # Calculate missing skills
    missing_skills = []
    for req_skill in all_required_skills:
        if not any(req_skill.lower() in found.lower() for found in standardized_found):
            missing_skills.append(req_skill)

    # Calculate skill match percentage
    if all_required_skills:
        matching_skills = [skill for skill in all_required_skills if skill not in missing_skills]
        skill_match_percentage = (len(matching_skills) / len(all_required_skills)) * 100
    else:
        skill_match_percentage = 0
        matching_skills = []

    # Holistic assessment
    holistic_analysis = analyze_holistic_skills(assessment_data, predicted_role)

    # Gap severity categorization
    gap_severity = categorize_gap_severity(skill_match_percentage)

    # Priority skills
    priority_skills = prioritize_missing_skills(missing_skills, predicted_role)

    # Learning path recommendations
    learning_path = generate_learning_path(priority_skills, predicted_role)

    # Generate recommendations
    recommendations = generate_personalized_recommendations(missing_skills, predicted_role, holistic_analysis)

    return {
        'technical_gaps': {
            'required_skills': all_required_skills,
            'found_skills': standardized_found,
            'missing_skills': missing_skills,
            'matching_skills': matching_skills,
            'skill_match_percentage': round(skill_match_percentage, 2),
            'priority_skills': priority_skills[:5]
        },
        'holistic_analysis': holistic_analysis,
        'gap_severity': gap_severity,
        'learning_path': learning_path,
        'recommendations': recommendations
    }


def generate_personalized_recommendations(missing_skills, predicted_role, holistic_analysis):
    """Generate personalized recommendations"""
    recommendations = {
        'immediate_actions': [],
        'skill_development': [],
        'career_advice': []
    }

    # Immediate actions based on skill gaps
    if len(missing_skills) > 15:
        recommendations['immediate_actions'].append(
            "Consider foundational courses in your target field before specializing"
        )
    elif len(missing_skills) > 10:
        recommendations['immediate_actions'].append(
            "Focus on 3-4 core skills first, then expand your skill set"
        )
    else:
        recommendations['immediate_actions'].append(
            "You're on the right track - focus on refining existing skills"
        )

    # Skill development recommendations
    critical_skills = {'python', 'sql', 'java', 'machine learning', 'data analysis'}
    missing_critical = [skill for skill in missing_skills if skill.lower() in critical_skills]

    if missing_critical:
        recommendations['skill_development'].append(
            f"Priority skills to develop: {', '.join(missing_critical[:3])}"
        )

    # Career advice based on holistic analysis
    if holistic_analysis['role_fit_score'] < 60:
        recommendations['career_advice'].append(
            "Consider exploring alternative career paths that better match your intelligence profile"
        )
    elif holistic_analysis['role_fit_score'] < 80:
        recommendations['career_advice'].append(
            "Your profile shows good potential - focus on developing specific intelligence areas"
        )

    return recommendations


# -----------------------------
# Helper Functions (existing)
# -----------------------------
def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.lower().strip()


def extract_text_from_file(filepath):
    if filepath.endswith('.pdf'):
        doc = fitz.open(filepath)
        return " ".join([page.get_text() for page in doc])
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


def extract_skills(text, required_skills):
    found_skills = []
    text_lower = text.lower()
    for skill in required_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.append(skill)
    return list(set(found_skills))


# -----------------------------
# Routes
# -----------------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin":
            return render_template('index.html')
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/assessment', methods=['GET', 'POST'])
def intelligence_assessment():
    """Route for holistic intelligence assessment"""
    if request.method == 'POST':
        # Store assessment responses in session
        assessment_data = {
            'linguistic': int(request.form.get('linguistic', 5)),
            'musical': int(request.form.get('musical', 5)),
            'logical_mathematical': int(request.form.get('logical_mathematical', 5)),
            'spatial': int(request.form.get('spatial', 5)),
            'bodily': int(request.form.get('bodily', 5)),
            'interpersonal': int(request.form.get('interpersonal', 5)),
            'intrapersonal': int(request.form.get('intrapersonal', 5)),
            'naturalist': int(request.form.get('naturalist', 5))
        }

        session['assessment_data'] = assessment_data
        return redirect(url_for('upload_resume'))

    return render_template('assessment.html')


@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        file = request.files['resume_file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract and clean resume text
            raw_text = extract_text_from_file(filepath)
            cleaned_text = clean_text(raw_text)

            # Predict job category
            vec = vectorizer.transform([cleaned_text])
            predicted_role = model.predict(vec)[0]

            # Extract found skills using existing logic
            required_skills = job_df[job_df["Job Role"] == predicted_role]["Required Skill"].tolist()
            found_skills = extract_skills(cleaned_text, required_skills)

            # Get assessment data if available
            assessment_data = session.get('assessment_data', None)

            # Comprehensive skill gap analysis
            analysis_results = comprehensive_skill_gap_analysis(
                predicted_role, found_skills, assessment_data
            )

            # Store results in session for detailed view
            session['analysis_results'] = analysis_results
            session['predicted_role'] = predicted_role
            session['raw_resume_text'] = raw_text[:500]  # Store first 500 chars for reference

            return render_template("result.html",
                                   category=predicted_role,
                                   analysis=analysis_results,
                                   career_path=career_recos.get(predicted_role, None),
                                   has_assessment=assessment_data is not None)

    return render_template("upload.html")


@app.route('/detailed_analysis')
def detailed_analysis():
    """Show detailed skill gap analysis"""
    analysis_results = session.get('analysis_results', {})
    predicted_role = session.get('predicted_role', '')

    if not analysis_results:
        return redirect(url_for('upload_resume'))

    return render_template('detailed_analysis.html',
                           analysis=analysis_results,
                           role=predicted_role)


@app.route('/learning_path')
def learning_path():
    """Show personalized learning path"""
    analysis_results = session.get('analysis_results', {})
    predicted_role = session.get('predicted_role', '')

    if not analysis_results:
        return redirect(url_for('upload_resume'))

    return render_template('learning_path.html',
                           learning_path=analysis_results.get('learning_path', {}),
                           role=predicted_role)


@app.route('/skill_recommendations')
def skill_recommendations():
    """Show skill-specific recommendations"""
    analysis_results = session.get('analysis_results', {})

    if not analysis_results:
        return redirect(url_for('upload_resume'))

    return render_template('skill_recommendations.html',
                           recommendations=analysis_results.get('recommendations', {}),
                           technical_gaps=analysis_results.get('technical_gaps', {}))


@app.route('/career_alternatives')
def career_alternatives():
    """Show alternative career suggestions"""
    analysis_results = session.get('analysis_results', {})

    if not analysis_results:
        return redirect(url_for('upload_resume'))

    holistic_analysis = analysis_results.get('holistic_analysis', {})

    return render_template('career_alternatives.html',
                           alternatives=holistic_analysis.get('alternative_careers', []),
                           current_fit=holistic_analysis.get('role_fit_score', 0))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
