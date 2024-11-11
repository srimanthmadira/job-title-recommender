import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

data_cleaned = pd.read_csv('job_data.csv')
data_cleaned['combined_features'] = data_cleaned['skills'].fillna('') + ' ' + data_cleaned['jobdescription'].fillna('') +  data_cleaned['jobtitle'].fillna('') 

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data_cleaned['combined_features'])

count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(data_cleaned['combined_features'])

def capitalize_skills(skills):
    return ', '.join(skill.strip().capitalize() for skill in skills.split(','))

def calculate_jaccard_similarity(input_skills, job_skills):
    """Calculate Jaccard Similarity."""
    input_set = set(input_skills.split(','))
    job_set = set(job_skills.split(','))
    intersection = input_set.intersection(job_set)
    union = input_set.union(job_set)
    return len(intersection) / len(union) if union else 0

def recommend_jobs(input_skills, threshold=25):

    input_vector_tfidf = tfidf_vectorizer.transform([input_skills])
    cosine_sim_tfidf = cosine_similarity(input_vector_tfidf, tfidf_matrix).flatten()

    input_vector_count = count_vectorizer.transform([input_skills])
    cosine_sim_count = cosine_similarity(input_vector_count, count_matrix).flatten()

    matched_jobs = []
    for idx, row in data_cleaned.iterrows():
        job_title = row['jobtitle']
        job_skills = row['skills'].lower()

        jaccard_sim = calculate_jaccard_similarity(input_skills, job_skills) * 100

        if cosine_sim_tfidf[idx] * 100 >= threshold or jaccard_sim >= threshold:
            matched_jobs.append({
                'jobtitle': job_title,
                'company': row['company'],
                'employment_type': row['employmenttype_jobstatus'],
                'city': row['city'],
                'state': row['state'],
                'cosine_sim_tfidf': round(cosine_sim_tfidf[idx] * 100, 2),
                'cosine_sim_count': round(cosine_sim_count[idx] * 100, 2),
                'jaccard_similarity': round(jaccard_sim, 2)
            })

    matched_jobs = sorted(matched_jobs, key=lambda x: x['cosine_sim_tfidf'], reverse=True)[:20]
    return matched_jobs

input_skills = input("Enter your skills (comma-separated): ").strip()

recommended_jobs = recommend_jobs(input_skills)

print(f"\nJob Recommendations for Skills: {capitalize_skills(input_skills)}\n")
for idx, job in enumerate(recommended_jobs, start=1):
    print(f"Job {idx}:")
    print(f"  Title: {job['jobtitle']}")
    print(f"  Company: {job['company']}")
    print(f"  Location: {job['city']}, {job['state']}")
    print(f"  Employment Type: {job['employment_type']}")
    print(f"  Cosine Similarity (TF-IDF): {job['cosine_sim_tfidf']}%")
    print(f"  Cosine Similarity (Raw Counts): {job['cosine_sim_count']}%")
    print(f"  Jaccard Similarity: {job['jaccard_similarity']}%")
    print("-" * 40)