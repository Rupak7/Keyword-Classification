from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pdftotext
import pandas as pd
import numpy as np
import sys
from nltk.stem import PorterStemmer

#nltk.download()
ps = PorterStemmer()
#coprora = body of text
#lexicon =  a dictionary i.e words and their means
#steming = find the root word

text = "The question of metrics, should be considered early, but this would also require a reference set of input item: a training set of sort, even though we are working off a pre-defined dictionary category-keywords (typically training sets are used to determine this very list of category-keywords, along with a weight factor). Of course such reference/training set should be both statistically significant and statistically representative [of the whole set]."

stop_words = set(stopwords.words("english"))

words = word_tokenize(text)

filtered_text=[]

for i in words:
    if i not in stop_words:
        filtered_text.append(i)

filter_txt_stem=[]
for i in filtered_text:
    filter_txt_stem.append(ps.stem(i))

print filter_txt_stem
