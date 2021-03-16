# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:49:29 2019

@author: bmwin
"""


from bs4 import BeautifulSoup
import urllib.request
from urllib.request import urlopen, Request
import pickle
from Levenshtein import distance
from tqdm import tqdm
import os
import pandas as pd
# import numpy as np
# import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn import metrics
# from sklearn.decomposition import LatentDirichletAllocation

corpora = {}
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'
}

years = list(range(1971, 2021, 1))
months = ['04', '10']
base_url = 'https://www.churchofjesuschrist.org'

for year in years:
    year_corpus = {}
    corpora[year] = year_corpus
    for month in months:
        year_corpus[month] = []
        req_url = '{}/study/general-conference/{}/{}?lang=eng'.format(base_url, year, month)
        req = Request(url=req_url, headers=headers) 
        #resp = urllib.request.urlopen("http://google.com/search?q=ford+stock")
        resp = urlopen(req)
        soup = BeautifulSoup(resp, from_encoding='UTF-8')
        page_links = soup.find_all('a', href=True)
        hrefs = [link['href'] for link in page_links]
        if (len(hrefs) > 0):
            for href in hrefs:
                try:
                    html = urllib.request.urlopen('{}{}'.format(base_url, href)).read()
                    soup = BeautifulSoup(html)
                        
                    text = soup.get_text()
                    
                    lines = (line.strip() for line in text.splitlines())
                    # break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # drop blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    text_length = len(text)
                    if text_length > 0:
                        print(href + ' size: ' + str(text_length))
                        year_corpus[month].append(text)
                except Exception as e: 
                    print('fail to get: ' + href + '\n' + str(e))

with open('corpora.pkl', 'wb') as f:
    pickle.dump(corpora, f)
    
with open('corpora.pkl', 'rb') as f:
    corpora = pickle.load(f)
    
#clean corpus
cleaned_corpora = {}
for year in years:
    print(str(year))
    year_corpus = corpora[year]
    cleaned_corpora[year] = {}
    for month in months:
        corpus = year_corpus[month]
        for i in tqdm(range(len(corpus)), total=len(corpus)):
            doc = corpus[i]
            doc_lines = []
            no_hits = 0
            for j in range(i+1, len(corpus)):
                if no_hits < 50:
                    other_doc = corpus[j]
                    for line in doc.split('\n'):
                        line_len = len(line)
                        for other_line in other_doc.split('\n'):
                            other_len = len(other_line)
                            size_diff = abs(line_len - other_len)
                            dist = distance(line, other_line)
                            if size_diff < 30 and dist < 10:
                                doc_lines.append(line)
                                break
                            else:
                                no_hits = no_hits + 1
                doc_lines = list(set(doc_lines))
                for line in doc_lines:
                    doc = doc.replace(line, '')
                corpus[i] = doc
        
        for i in tqdm(range(len(corpus)), total=len(corpus)):
            doc = corpus[i]
            doc = os.linesep.join([s for s in doc.splitlines() if s])
            corpus[i] = doc
            
        cleaned = [doc for doc in corpus if len(doc) > 200]
        
        #process with regex to fix some stuff
        import re
        cleaned = [re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', doc) for doc in cleaned]
        cleaned = [re.sub(r'(?<=\.)(?=[A-Z])', ' ', doc) for doc in cleaned]
        cleaned = [re.sub(r'(?<=\d)(?=[A-Z])', ' ', doc) for doc in cleaned]
        cleaned_corpora[year][month] = cleaned

with open('cleaned_corpora.pkl', 'wb') as f:
    pickle.dump(cleaned_corpora, f)
    
with open('cleaned.pkl', 'rb') as f:
    cleaned_corpora = pickle.load(f)
    
#preprocess
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

class Tokenizer(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.NLTKWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self,text):
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokenized_sents = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.lemmatizer.lemmatize(tok.lower()) for tokens in tokenized_sents for tok in tokens if tok not in stop_words and tok.isalpha()]
        return lemmas
    
tokenizer = Tokenizer()
preprocessed_corpora = {}
for year in tqdm(years, total=len(years)):
    preprocessed_corpora[year] = {}
    for month in months:
        cleaned = cleaned_corpora[year][month]
        preprocessed_corpora[year][month] = [tokenizer.tokenize(doc) for doc in cleaned]

with open('preprocessed_corpora.pkl', 'wb') as f:
    pickle.dump(preprocessed_corpora, f)
    
with open('preprocessed_corpora.pkl', 'rb') as f:
    preprocessed_corpora = pickle.load(f)
    
#N-grams
preprocessed = [toks for year,v in preprocessed_corpora.items() for month,p in v.items() for toks in p]

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary

bigram_model = Phrases(preprocessed, min_count=5, threshold=10)
trigram_model = Phrases(bigram_model[preprocessed], min_count=5, threshold=10)

bigram_phraser = Phraser(bigram_model)
trigram_phraser = Phraser(trigram_model)

ngram_docs = []
for doc in tqdm(preprocessed, total=len(preprocessed)):
    bigrams = bigram_phraser[doc]
    trigrams = trigram_phraser[bigrams]
    ngram_docs.append(trigrams)
    
dictionary = Dictionary(ngram_docs)
dictionary.filter_extremes(no_below=3, no_above=0.8, keep_n=100000)
bow_docs = [dictionary.doc2bow(doc) for doc in ngram_docs]

with open('bow_and_dict.pkl', 'wb') as f:
    pickle.dump((bow_docs, dictionary, ngram_docs), f)
    
with open('bow_and_dict.pkl', 'rb') as f:
    bow_docs, dictionary, ngram_docs = pickle.load(f)
    
#topic modeling
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import CoherenceModel

path_to_mallet_binary = 'd:/mallet-2.0.8/bin/mallet'
output_path = 'd:/code/gc_text_analysis/mallet_output/'
num_topics=140
model = LdaMallet(path_to_mallet_binary, 
                  corpus=bow_docs, 
                  workers=4,
                  iterations=2000,
                  num_topics=num_topics, 
                  id2word=dictionary, 
                  prefix=output_path)

model.save('gc_lda_model.pkl')

dictionary.id2token = dict((v,k) for k,v in dictionary.token2id.items())
words_freq = [(dictionary.id2token[id], cnt) for id, cnt in dictionary.dfs.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq = pd.DataFrame(words_freq, columns=['word', 'count'])

coherence_model_lda = CoherenceModel(model=model, texts=ngram_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

topics = model.show_topics(num_topics=num_topics, num_words=10, log=False, formatted=False)
topics = list(zip(*topics))[1]

gc_topics = model[bow_docs[-73:]]