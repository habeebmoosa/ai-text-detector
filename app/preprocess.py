import pandas as pd

def preprocess_text(text):
    # Create a dummy DataFrame with 'text' column
    df = pd.DataFrame({'text': [text]})
    
    # Apply dummy functions to mimic preprocessing
    df['text'] = df['text'].apply(dummy_space_remover)
    df['cleaned_text'] = df['text'].apply(dummy_clean_text)

    return df

def dummy_space_remover(text):
    # Dummy function: removes spaces
    return text.strip()

def dummy_clean_text(text):
    # Dummy function: lowercase and remove non-alphabetic characters
    return text.lower().replace('.', '')

def postprocess_text(df):
    duplicate_columns = df.columns.duplicated(keep='first')

    if any(duplicate_columns):
        df.columns = [f"{col}.1" if duplicate else col for col, duplicate in zip(df.columns, duplicate_columns)]

    columns_to_drop = [col for col in df.columns if col.endswith('.1')]

    df = df.drop(columns=columns_to_drop)

    return df