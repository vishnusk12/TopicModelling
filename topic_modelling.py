# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 10:39:06 2017

@author: Vishnu
"""

from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import gensim
wnl = WordNetLemmatizer()

def preprocess(raw_text):
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    letters_only_text = wnl.lemmatize(letters_only_text)
    words = letters_only_text.lower().split()
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w.isalpha()]
    meaningful_words = [w for w in words if w not in stopword_set]
    cleaned_word_list = " ".join(meaningful_words)
    return cleaned_word_list

def LDA(text):
    text = preprocess(text)
    texts = [text.split()]
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(i) for i in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word = dictionary, passes=20)
    X = ldamodel.print_topics(num_topics=1, num_words=30)
    return X