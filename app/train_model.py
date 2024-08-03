import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PyPDF2 import PdfFileReader
import os

# Load the CSV file
data = pd.read_csv('path_to_your_csv_file.csv')

# Function to read PDF files
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Add the text from PDF to the dataframe
def add_pdf_text(row):
    pdf_path = f'data/{row["Category"]}/{row["ID"]}.pdf'
    if os.path.exists(pdf_path):
        return read_pdf(pdf_path)
    return ''

data['Resume_text'] = data.apply(lambda row: row['Resume_str'] + ' ' + add_pdf_text(row), axis=1)

# Prepare features and labels
X = data['Resume_text']
y = data['Category']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with a TF-IDF vectorizer and a logistic regression model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(pipeline, 'resume_model.pkl')
