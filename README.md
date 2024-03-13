# AI Generated Text Detection

This project aims to classify text as either human-generated or AI-generated. It utilizes a variety of natural language processing (NLP) features and machine learning algorithms to achieve this classification task.

## Features
The following features are extracted from the provided dataset:
- **Basic NLP Features:**
  - Char count, word count, word density, punctuation count, title word count, upper-case count, noun count, adverb count, verb count, adjective count, pronoun count.
- **Term Frequencies and N-gram:**
  - Count vectorizer with 35742 features.
  - Bigram words (5000 features).
  - Trigram words (5000 features).
  - BiTrigram characters (5000 features).
- **Topic Modeling:**
  - NeuralLDA with 20 topics.
- **Others:**
  - Readability score, Named Entity Recognition (NER) count, text error length, and Lexical Diversity.

## Feature Selection
After feature extraction, Principal Component Analysis (PCA) is applied with `n_components` set to 256 for feature selection.

## Algorithms
The project utilizes five different algorithms for training and testing:
1. Random Forest
2. Support Vector Machine (SVM)
3. XGBoost
4. Gradient Boosting
5. Logistic Regression

## Performance
Among the five algorithms tested, Gradient Boosting demonstrated superior performance. It provided accurate classification results during the prediction phase.

## Flask Application
A simple Flask application is developed to demonstrate the functionality of the AI Generated Text Detection model. Users can input text, and the application will classify it as either human-generated or AI-generated.

## Usage
To use the project:
1. Clone the repository from GitHub.
2. Install the required dependencies.
3. Run the Flask application.
4. Input text to classify whether it is human-generated or AI-generated.

## Contributors
- [Your Name](https://github.com/yourusername) - Project Lead & Developer

## License
This project is licensed under the [MIT License](LICENSE).
