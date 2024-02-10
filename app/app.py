from flask import Flask, request, jsonify, render_template
from preprocess import preprocess_text
from features import feature_extraction
import joblib

app = Flask(__name__)

def predict_text(df):
    model = joblib.load('model/rf_model_v3.pkl')

    X = df.drop(['text', 'cleaned_text'], axis=1)

    prediction = model.predict(X)

    return prediction

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['inputbox']
    process_data = preprocess_text(text)
    result_df = feature_extraction(process_data)
    prediction = predict_text(result_df)

    if prediction == 0:
        return jsonify({'result': 'The text likely to be human written'})
    return jsonify({'result': 'The text likely to be AI Generated'})

if __name__ == '__main__':
    app.run(debug=True)