# import pandas as pd
# from nltk.tokenize import word_tokenize
# import string
# from nltk import pos_tag, ne_chunk
# import textstat
# from language_tool_python import LanguageTool
import pandas as pd
import string
import joblib

def feature_extraction(df):
    count_vectorizer = joblib.load('tools/count_vectorizer_50k.pkl')
    bigram_vectorizer = joblib.load('tools/bigram_vectorizer_50k.pkl')
    trigram_vectorizer = joblib.load('tools/trigram_vectorizer_50k.pkl')
    bitri_vectorizer = joblib.load('tools/bitri_vectorizer_50k.pkl')

    # Basic NLP ------------------------------------------------

    df['char_count'] = 0  # Dummy value

    df['word_count'] = 0  # Dummy value

    df['word_density'] = 0  # Dummy value

    df['punctuation_count'] = 0  # Dummy value

    df['upper_case_count'] = 0  # Dummy value

    df['title_word_count'] = 0  # Dummy value

    df[['noun_count','adv_count','verb_count','adj_count','pro_count']] = 0  # Dummy value

    # Topic Modeling -------------------------------------------------------

    num_topics = 20
    for topic in range(num_topics):
        df[f'topic_{topic + 1}_score'] = 0  # Dummy value

    # Readability Scores -------------------------------------------------------

    df['flesch_kincaid_score'] = 0  # Dummy value

    df['flesch_score'] = 0  # Dummy value

    df['gunning_fog_score'] = 0  # Dummy value

    df['coleman_liau_score'] = 0  # Dummy value

    df['dale_chall_score'] = 0  # Dummy value

    df['ari_score'] = 0  # Dummy value

    df['linsear_write_score'] = 0  # Dummy value

    df['spache_score'] = 0  # Dummy value

    # Named entity recognition ----------------------------------------------------
    df['ner_count'] = 0  # Dummy value

    # Text error length -----------------------------------------------------------
    df['error_length'] = 4  # Dummy value

    # Preprocessing again --------------------------------------------------------------

    df.rename(columns={'text': 'normal_text'}, inplace=True)

    # count vectorization --------------------------------------------------------------

    count_matrix = count_vectorizer.transform(df['cleaned_text'])

    feature_names = count_vectorizer.get_feature_names_out()
    count_df = pd.DataFrame(count_matrix.toarray(), columns=feature_names)
    df = pd.concat([df, count_df], axis=1)
    count_df = pd.DataFrame(count_matrix, columns=['dummy_feature'])
    df = pd.concat([df, count_df], axis=1)

    # bi gram vectorization --------------------------------------------------------------

    bigram_matrix = bigram_vectorizer.transform(df['cleaned_text'])
    bigram_feature_names = bigram_vectorizer.get_feature_names_out()
    bigram_df = pd.DataFrame(bigram_matrix.toarray(), columns=bigram_feature_names)

    df = pd.concat([df, bigram_df], axis=1)

    # tri gram vectorization --------------------------------------------------------------
    trigram_matrix = trigram_vectorizer.transform(df['cleaned_text'])
    trigram_feature_names = trigram_vectorizer.get_feature_names_out()
    trigram_df = pd.DataFrame(trigram_matrix.toarray(), columns=trigram_feature_names)

    df = pd.concat([df, trigram_df], axis=1)

    # bi tri gram vectorization --------------------------------------------------------------

    bichar_matrix = bitri_vectorizer.transform(df['cleaned_text'])

    bichar_df = pd.DataFrame(bichar_matrix.toarray(), columns=bitri_vectorizer.get_feature_names_out())

    bichar_df.columns = bichar_df.columns.str.strip()

    bichar_df = bichar_df.loc[:, ~bichar_df.columns.duplicated()]

    df = pd.concat([df, bichar_df], axis=1)

    # --------------------------------------------------------------------------------------------

    ttr_list = [0] * len(df)  # Dummy value
    ttr_df = pd.DataFrame({'lexical_diversity': ttr_list})
    df['lexical_diversity'] = ttr_df['lexical_diversity']

    return df


# Example usage
# df = pd.DataFrame({'cleaned_text': ["Your cleaned text here"]})
# feature_extraction(df)
