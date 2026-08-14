# AI Salary Prediction System
**AI-powered salary prediction web application**
<img width="1495" height="813" alt="Image" src="https://github.com/user-attachments/assets/2953275f-efc1-4326-8350-7fcaaff20764" />
<img width="1447" height="832" alt="Image" src="https://github.com/user-attachments/assets/f371cc64-0a56-450f-b418-d0ddcb64fd07" />
An AI-powered salary prediction web application that estimates salaries for job roles using real-world Indian job market data.
The system uses machine learning to analyze job title, skills, location, and years of experience and predict an estimated annual salary.

## Project Overview

Salary prediction is a regression problem where multiple factors influence the expected salary of a job.

This project uses a real-world Indian job market dataset containing approximately 98,000 job postings.

The model is trained only on records where salary information is available and uses:

- Job Title
- Skills
- Job Location
- Years of Experience

The application provides an estimated annual salary, monthly salary, and salary range through a Flask web interface.

## Features

- Real-world Indian job market dataset
- Machine learning salary prediction
- Job role selection
- Location selection
- Skills input
- Experience-based prediction
- Estimated annual salary
- Estimated monthly salary
- Estimated salary range
- Skill recommendations for selected job roles
- Flask-based web application
- Trained machine learning model

## Dataset

The project uses the:

`indian-job-market-dataset-2025.xlsx`

Dataset size:

- Total job records: 97,929
- Usable INR salary records: 33,209
- Records after salary outlier handling: 33,059

Salary information is available as minimum and maximum salary values.

The target variable is calculated as:

```text
Average Salary = (Minimum Salary + Maximum Salary) / 2
