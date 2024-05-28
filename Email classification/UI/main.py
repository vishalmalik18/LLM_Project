from flask import Flask, request,jsonify
import joblib
from helper import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("D:/NLP projects/LLM_Project/Email classification/model.pkl")
vectorizer = joblib.load("D:/NLP projects/LLM_Project/Email classification/vector.pkl")

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():

    email_text = request.json.get('email_text')

    if not email_text:
        return jsonify({
            'error':'No email text provided'
        })

    cleaning_area = [preprocessing(n) for n in email_text]

    convert_into_vector = vectorizer.transform(cleaning_area)

    predication = model.predict(convert_into_vector)

    return jsonify({
        'predication':predication.tolist()
    })

if __name__ =='__main__':
    app.run(debug=True)