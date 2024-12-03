from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO
# Initialize Flask app
app = Flask(__name__)


# Google Drive link
drive_url = "https://drive.google.com/uc?id=1jPjsVx6Qg096VcWYvBQT3qzc2RmY3pA7"

# Read dataset directly into a DataFrame
response = requests.get(drive_url)
data_cleaned = pd.read_csv(StringIO(response.text))



data_cleaned['combined_features'] = data_cleaned['skills'].fillna('') + ' ' + data_cleaned['jobdescription'].fillna('')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data_cleaned['combined_features'])


# Utility function to capitalize skills
def capitalize_skills(skills):
    return ', '.join(skill.strip().capitalize() for skill in skills.split(','))


# Recommendation Function
def recommend_jobs(input_skills, threshold=25):
    input_skills_list = {skill.strip().lower() for skill in input_skills.split(',')}
    input_vector = vectorizer.transform([input_skills])
    cosine_sim = cosine_similarity(input_vector, tfidf_matrix)

    matched_jobs = []
    for idx, row in data_cleaned.iterrows():
        job_title = row['jobtitle']
        job_skills = row['skills'].lower().split(',')
        job_skills = [skill.strip() for skill in job_skills]

        matching_skills = input_skills_list.intersection(job_skills)
        user_match_percentage = (len(matching_skills) / len(input_skills_list)) * 100 if input_skills_list else 0
        required_match_percentage = (len(matching_skills) / len(job_skills)) * 100 if job_skills else 0
        other_required_skills = set(job_skills) - matching_skills

        if user_match_percentage >= threshold:
            matched_jobs.append({
                'jobtitle': job_title,
                'company': row['company'],
                'employment_type': row['employmenttype_jobstatus'],
                'city': row['city'],
                'state': row['state'],
                'user_match_percentage': round(user_match_percentage, 2),
                'required_match_percentage': round(required_match_percentage, 2),
                'other_required_skills': capitalize_skills(', '.join(other_required_skills))
            })

    # Sort by required skill match percentage and take the top 20
    matched_jobs = sorted(matched_jobs, key=lambda x: x['required_match_percentage'], reverse=True)[:20]
    recommended_jobs = pd.DataFrame(matched_jobs).drop_duplicates(subset=['jobtitle'])
    return recommended_jobs


@app.route('/')
def home():
    return render_template('home.html')


# Routes
@app.route('/recommendation', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_skills = request.form['skills']
        recommended_jobs = recommend_jobs(input_skills)
        input_skills_capitalized = ', '.join(skill.strip().capitalize() for skill in input_skills.split(','))

        return render_template('index_recommendation_page.html', recommended_jobs=recommended_jobs.to_dict(orient='records'),
                               input_skills=input_skills_capitalized)
    
    return render_template('index_recommendation_page.html', recommended_jobs=None, input_skills=None)


@app.route('/insights', methods=['GET'])
def latest_trends():
    # Data aggregation for trends
    city_job_counts = data_cleaned['city'].value_counts().head(7).to_dict()
    state_job_counts = data_cleaned['state'].value_counts().head(5).to_dict()

    # Exclude empty skills from top_skills calculation
    top_skills = data_cleaned['skills'].str.split(',').explode().str.strip()
    top_skills = top_skills[top_skills != ''].value_counts().head(10).to_dict()

    most_common_jobtitles = data_cleaned['jobtitle'].value_counts().head(10).to_dict()

    return render_template('index_insights_page.html',
                           top_7_cities=city_job_counts,
                           top_5_states=state_job_counts,
                           top_skills=top_skills,
                           most_common_jobtitles=most_common_jobtitles)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
