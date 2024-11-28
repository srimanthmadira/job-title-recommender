# job-title-recommender
## Overview
The Job Recommendation System is a Flask-based web application that helps users find job roles suited to their skill set. By leveraging natural language processing techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity, the application matches users’ inputted skills with job descriptions in a pre-existing dataset. The app provides users with a personalized list of recommended jobs, along with skill matching percentages, enabling users to easily identify job roles that best align with their expertise.

## Key Components
**1. **Flask Backend (app.py):**** The app.py file serves as the main application entry point, utilizing Flask to handle user requests and render HTML templates. It contains the main logic for processing user inputs, running the recommendation algorithm, and serving the results back to the user.

**2. Data Processing and Skill Matching:** The backend reads from a job dataset (job_data.csv), which contains fields such as jobtitle, jobdescription, and skills. The system uses TF-IDF vectorization to represent both the user input and job descriptions as numerical vectors. Cosine similarity is then applied to measure the closeness between user skills and job descriptions, providing a percentage match for each job.

**3. Recommendation Algorithm:** The core recommendation logic involves:
- Parsing the user’s inputted skills and comparing them against job descriptions.
- Calculating match percentages based on the overlap of user and job skills.
- Returning jobs that exceed a defined match threshold, ensuring relevance to the user’s profile.
  
**4. User Interface:** The interface is simple and intuitive, with an HTML form that lets users enter their skills and see recommended jobs instantly. The UI is styled with CSS to improve readability and usability, offering a clean and professional experience.

**5. Deployment and Execution:** The application is designed to run locally, with a virtual environment set up to manage dependencies. Flask runs the app on a local server, allowing users to interact with it through a web browser.

## Technical Stack
- **Python:** The primary language used for backend development and data processing.
- **Flask:** A micro web framework used to handle HTTP requests, render HTML pages, and manage the application's routing.
- **Pandas:** Utilized for data manipulation and handling the job dataset.
- **Scikit-learn:** Provides the TF-IDF vectorizer and cosine similarity functions for skill matching.
- **HTML/CSS:** HTML templates and CSS styling for the frontend user interface.

## Features and Functionality
- **Skill-Based Job Matching:** Users input their skills, and the app recommends jobs based on skill similarity.
- **Skill Match Percentage:** Each job recommendation comes with a skill match percentage, giving users a sense of how well their skills align with the job requirements.
- **Additional Skill Insights:** Along with recommended jobs, the app displays any additional skills required for each role, allowing users to identify skill gaps.

# Preview 
![image](https://github.com/user-attachments/assets/03d7c564-d9b7-4671-a214-d5175f04d888)
![image](https://github.com/user-attachments/assets/852d9550-1a55-4281-90be-d49847cb4bc8)
![image](https://github.com/user-attachments/assets/3a1e5797-58db-4b59-a082-a06fa2b9c7d4)




