import re
import gensim
from gensim.models import Word2Vec

import nltk
from nltk.corpus import brown
from nltk.corpus import abc


import nltk
nltk.download('brown')
nltk.download('abc')
nltk.download('punkt')

def corpus_normalise(corpus):
    corpus_norm = []
    for sent in corpus.sents():
        sent = re.sub('[^a-z0-9]+', ' ', ' '.join(sent).lower()) # lowercase, remove punctuation & non-alphanumeric characters
        corpus_norm.append(sent.strip().split())
    return corpus_norm

corpus_norm = corpus_normalise(abc)
                      
for sentence in corpus_norm[:5]: # show the first 5 sentences of the corpus
    print(sentence)

