import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

# Load or train your model here
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
model = joblib.load('path_to_your_model.pkl')

def classify_resume(resume, job_desc):
    vectorized_text = vectorizer.transform([resume])
    return model.predict(vectorized_text)[0]
