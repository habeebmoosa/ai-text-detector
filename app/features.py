import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk import pos_tag, ne_chunk
from gensim import corpora
from gensim.models import LdaModel
import textstat
from language_tool_python import LanguageTool

def feature_extraction(df):
    df['char_count'] = df['cleaned_text'].apply(len)

    df['word_count'] = df['cleaned_text'].apply(lambda x: len(word_tokenize(x))) # word count

    df['word_density'] = df['word_count'] / df['char_count']

    df['punctuation_count'] = df['text'].apply(punctuation_count)

    df['upper_case_count'] = df['text'].apply(upper_case_count)

    df['title_word_count'] = df['text'].apply(title_word_count)

    df[['noun_count','adv_count','verb_count','adj_count','pro_count']] = df['cleaned_text'].apply(lambda x: parts_of_speech(x))

    # Topic Modeling

    corpus = [text.split() for text in df['cleaned_text']]

    dictionary = corpora.Dictionary(corpus)

    corpus_bow = [dictionary.doc2bow(text) for text in corpus]

    # Training
    num_topics = 20
    lda_model = LdaModel(corpus_bow, num_topics=num_topics, id2word=dictionary, passes=15)

    topic_distribution = lda_model.get_document_topics(corpus_bow)

    for topic in range(num_topics):
        df[f'topic_{topic + 1}_score'] = [next((t[1] for t in topic_dist if t[0] == topic), 0) for topic_dist in topic_distribution]

    # Readability Scores

    df['flesch_kincaid_score'] = df['cleaned_text'].apply(lambda x: textstat.flesch_kincaid_grade(x))

    df['flesch_score'] = df['cleaned_text'].apply(lambda x: textstat.flesch_reading_ease(x))

    df['gunning_fog_score'] = df['cleaned_text'].apply(lambda x: textstat.gunning_fog(x))

    df['coleman_liau_score'] = df['cleaned_text'].apply(lambda x: textstat.coleman_liau_index(x))

    df['dale_chall_score'] = df['cleaned_text'].apply(lambda x: textstat.dale_chall_readability_score(x))

    df['ari_score'] = df['cleaned_text'].apply(lambda x: textstat.automated_readability_index(x))

    df['linsear_write_score'] = df['cleaned_text'].apply(lambda x: textstat.linsear_write_formula(x))

    df['spache_score'] = df['cleaned_text'].apply(lambda x: textstat.spache_readability(x))

    df['ner_count'] = df['text'].apply(ner_count)

    df['error_length'] = df['text'].apply(error_length)

    return df


def punctuation_count(text):
    return sum(1 for char in text if char in string.punctuation)

def upper_case_count(text):
    return sum(1 for char in text if char.isupper())

def title_word_count(text):
    return sum(1 for word in text.split() if word.istitle())

def parts_of_speech(text):
    pos_tags = pos_tag(word_tokenize(text))
    
    noun_count = sum(1 for tag in pos_tags if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS'])
    adv_count = sum(1 for tag in pos_tags if tag[1] in ['RB', 'RBR', 'RBS'])
    verb_count = sum(1 for tag in pos_tags if tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
    adj_count = sum(1 for tag in pos_tags if tag[1] in ['JJ', 'JJR', 'JJS'])
    pro_count = sum(1 for tag in pos_tags if tag[1] in ['PRP', 'PRP$', 'WP', 'WP$'])
    return pd.Series([noun_count, adv_count, verb_count, adj_count, pro_count], index=['noun_count','adv_count','verb_count','adj_count','pro_count'])

def ner_count(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    ner_tags = ne_chunk(pos_tags)
    ner_count = sum(1 for chunk in ner_tags if hasattr(chunk, 'label'))
    return ner_count

def error_length(text):
    tool = LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)