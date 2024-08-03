from flask import render_template, request
from app import app
from app.model import classify_resume

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        resume = request.files['resume'].read().decode('utf-8')
        job_desc = request.files['job_description'].read().decode('utf-8')
        prediction = classify_resume(resume, job_desc)
        return f"Resume is {'relevant' if prediction == 1 else 'non-relevant'}"
