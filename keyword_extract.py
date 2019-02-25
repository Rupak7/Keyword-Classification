import pandas as pd
import numpy as np
import sys
from collections import Counter
import string
from tqdm import tqdm

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tag import pos_tag


def remove_stopwords(words):
    for stopword in stop_words:
        if stopword in words:
            words = list(filter(lambda a: a != stopword, words))

    return words

def calculate_tf_idf(raw_count, max_raw_count_in_document, no_documents, no_documents_in_which_word_occured):
    tf = 0.5 + 0.5*(raw_count/max_raw_count_in_document)
    idf = np.log(no_documents/(1 + no_documents_in_which_word_occured))
    return tf*idf

stop_words = set(stopwords.words("english"))

def userkeyword(user_input):
    document_text = list(user_input.split(" "))
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer('[A-Z][a-z]\w+')

    print("Generating Keywords ..")

    # Tokenizing and collecting all NNPs (Proper Nouns)
    for i in tqdm(range(len(document_text)), desc='Tokenizing: '):
        document_text[i] = pos_tag(word_tokenize(document_text[i]))
        document_text[i] = [word[0] for word in document_text[i] if word[1].startswith("VB") or word[1].startswith("NN") or word[1].startswith("JJ")]
        document_text[i] = [word.lower() for word in document_text[i]]

    # Remove all Stop words
    for i in tqdm(range(len(document_text)), desc='Cleaning: '):
        document_text[i] = remove_stopwords(document_text[i])

    # Lemmatizing
    #for i in range(len(document_text)):
    #    document_text[i] = [lemmatizer.lemmatize(word) for word in document_text[i]]
    for i in tqdm(range(len(document_text)), desc='Stemming: '):
        document_text[i] = [stemmer.stem(word) for word in document_text[i]]

    total_text = []

    for text in document_text:
        total_text += list(text)

    # Raw Count for all keywords
    word_count = Counter(total_text)
    raw_count = Counter(total_text)
    max_raw_count_in_document = next(iter(word_count.values()))

    # Calculating TF-IDF Values for all keywords
    for word in tqdm(word_count, desc='Calculating TF-IDF: '):
        count = 0
        for text in document_text:
            if word in text:
                count += 1

        no_documents_in_which_word_occured = count
        word_count[word] = calculate_tf_idf(word_count[word], max_raw_count_in_document, len(document_text), no_documents_in_which_word_occured)

    word_count = list(word_count.items())

    i = 0
    for w in raw_count:
        word_count[i] = word_count[i] + (raw_count[w], )
        i += 1

    # Storing all the keywords in Data-Frame
    df = pd.DataFrame(list(word_count), columns=['Keywords', 'TF_IDF', "Raw_Count"])

    df = df.sort_values('TF_IDF', ascending=True)

    cleaned_input = []
    for i in df.Keywords:
        cleaned_input.append(i)

    str1 = ' '.join(cleaned_input)
    return str1

print (userkeyword("I have stomach pain"))
