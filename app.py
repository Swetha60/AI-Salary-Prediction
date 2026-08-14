from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "model/salary_prediction_model.pkl"
)


# ============================================================
# LOAD DATA FOR DROPDOWN OPTIONS
# ============================================================

df = pd.read_excel(
    "data/indian-job-market-dataset-2025.xlsx"
)


job_roles = sorted(
    df["title"]
    .dropna()
    .unique()
    .tolist()
)


locations = sorted(
    df["location"]
    .dropna()
    .unique()
    .tolist()
)

# ============================================================
# RECOMMENDED SKILLS
# ============================================================

skill_recommendations = {

    "Data Engineer": [
        "Python",
        "SQL",
        "Spark",
        "Hadoop",
        "AWS",
        "Airflow"
    ],

    "Data Scientist": [
        "Python",
        "SQL",
        "Machine Learning",
        "Pandas",
        "NumPy",
        "Scikit-learn"
    ],

    "Machine Learning Engineer": [
        "Python",
        "Machine Learning",
        "TensorFlow",
        "PyTorch",
        "SQL",
        "Docker"
    ],

    "Python Developer": [
        "Python",
        "Django",
        "Flask",
        "SQL",
        "REST API",
        "Git"
    ],

    "Java Developer": [
        "Java",
        "Spring Boot",
        "SQL",
        "Microservices",
        "Git",
        "REST API"
    ],

    "Web Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "SQL"
    ],

    "Software Engineer": [
        "Python",
        "Java",
        "SQL",
        "Git",
        "Data Structures",
        "Algorithms"
    ]
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        job_roles=job_roles,
        locations=locations,
        skill_recommendations=skill_recommendations
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    job_role = request.form["job_role"]

    skills = request.form["skills"]

    location = request.form["location"]

    experience = float(
        request.form["experience"]
    )


    # Create input DataFrame
    input_data = pd.DataFrame(
        {
            "title": [job_role],

            "tagsAndSkills": [skills],

            "location": [location],

            "experienceYears": [experience]
        }
    )


    # ========================================================
    # PREDICT SALARY
    # ========================================================

    predicted_salary = model.predict(
        input_data
    )[0]


    # Prevent negative salary
    predicted_salary = max(
        predicted_salary,
        0
    )


    # ========================================================
    # ESTIMATED SALARY RANGE
    # ========================================================

    lower_salary = predicted_salary * 0.85

    upper_salary = predicted_salary * 1.15


    # ========================================================
    # MONTHLY SALARY
    # ========================================================

    monthly_salary = predicted_salary / 12

    lower_monthly = lower_salary / 12

    upper_monthly = upper_salary / 12


    # ========================================================
    # SEND RESULT TO HTML
    # ========================================================

    return render_template(
        "result.html",

        job_role=job_role,

        skills=skills,

        location=location,

        experience=experience,

        predicted_salary=predicted_salary,

        monthly_salary=monthly_salary,

        lower_salary=lower_salary,

        upper_salary=upper_salary,

        lower_monthly=lower_monthly,

        upper_monthly=upper_monthly
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )