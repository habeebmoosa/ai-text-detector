from flask import Flask, request, jsonify, render_template
from preprocess import preprocess_text, postprocess_text
from features import feature_extraction
import joblib
import nltk

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

app = Flask(__name__, static_url_path='/static')

def predict_text(df):
    model = joblib.load('model/gb_model_v1.pkl')
    pca = joblib.load('model/pca_v5.pkl')

    X = df.drop(['normal_text', 'cleaned_text'], axis=1)
    X_pca = pca.transform(X)

    prediction_proba = model.predict_proba(X_pca)
    prediction = model.predict(X_pca)

    return prediction, prediction_proba

@app.route('/')
def detect():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['inputbox']
    
    process_data = preprocess_text(text)

    feature_df = feature_extraction(process_data)

    result_df = postprocess_text(feature_df)

    prediction, prediction_proba = predict_text(result_df)

    probability_human = round(prediction_proba[0][0] * 100, 2)
    probability_ai = round(prediction_proba[0][1] * 100, 2)

    if prediction == 0:
        result = "human"
    else:
        result = "ai"

    return jsonify({'result': result, 'probability1': str(probability_human), 'probability2': str(probability_ai)})