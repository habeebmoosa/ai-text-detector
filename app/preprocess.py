import pandas as pd
import re
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    df = pd.DataFrame({'text': [text]})
    df['text'] = df['text'].apply(space_remover)
    df['cleaned_text'] = df['text'].apply(clean_text)

    return df

def space_remover(text):
    words = word_tokenize(text)
    text = ' '.join(words)
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    words = word_tokenize(text)

    cleaned_text = ' '.join(words)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text