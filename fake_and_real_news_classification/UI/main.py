from flask import Flask , request ,jsonify
import joblib
from helper import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer 

model = joblib.load("D:/NLP projects/model.pkl")

vectorizer = joblib.load("D:/NLP projects/vector.pkl")

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():

    news_text = request.json.get('news_text')

    if not news_text:
        return jsonify({'error': 'No news text provided'})

    cleaning_area = [preprocessing(n) for n in news_text]

    convert_into_vector  = vectorizer.transform(cleaning_area)

    predication = model.predict(convert_into_vector)


    return jsonify({
        'prediction':predication.tolist()
    })
    


if __name__ == '__main__':
    app.run(debug=True)
