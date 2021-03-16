# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:58:42 2021

@author: bmwin
"""

import pandas as pd
import numpy as np
import gensim
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import CoherenceModel
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from multiprocessing import Pool, cpu_count
import multiprocessing
import itertools
from pathlib import Path


train = False
model_version = 9
uuid = '42b43ad2-8c2e-4bdc-ba0e-9c31060d4d3f'
min_origination_date = 20060000000000

path_to_mallet_binary = 'f:/mallet-2.0.8/bin/mallet.bat'
path_to_mallet_output = 'f:/code/CAP/keywords/mallet_output/v{}/'.format(model_version)
bow_dictionary_file = '{}/bow_corpus_and_dictionary.pickle'.format(path_to_mallet_output)

text_fields = [
    'Title',
    'Condition',
    # 'ActionTaken',
    'OperationalNotes',
    'IssueNotes',
    # 'Recommendations',
    # 'EquipmentDescription'
    ]

stop_words = stopwords.words('english')
tokenizer = CountVectorizer(token_pattern=r'([a-zA-Z]{2,})').build_tokenizer()
lemmatizer = WordNetLemmatizer()
lemmatized_stop_words = [lemmatizer.lemmatize(word) for word in stop_words]

def preprocess(batch):
    result = []
    for text in batch:
        text_tokens = []
        for token in tokenizer(text):
            lemma = lemmatizer.lemmatize(token.lower())
            if lemma not in lemmatized_stop_words:
                text_tokens.append(lemma)
        result.append(text_tokens)
    return result

def generateNGrams(docs):
    bigram_model, trigram_model = build_ngram_model(docs)
   
    docs = [ bigram_model[doc] for doc in docs ]
    docs = [ trigram_model[bigram_model[doc]] for doc in docs ]
    return docs

def build_ngram_model(docs):
    bigram_model_path = Path('bigram_phraser.pkl')
    trigram_model_path = Path('trigram_phraser.pkl')
    if not bigram_model_path.exists() or not trigram_model_path.exists():
        print('Building n-gram models')
        bigram = Phrases(docs, min_count=3, threshold=6)
        trigram = Phrases(bigram[docs], min_count=3, threshold=6)
       
        bigram_model = Phraser(bigram)
        trigram_model = Phraser(trigram)
       
        bigram_model.save('bigram_phraser.pkl')
        trigram_model.save('trigram_phraser.pkl')
    else:
        print('Loading saved n-gram models')
        bigram_model = Phraser.load('bigram_phraser.pkl')
        trigram_model = Phraser.load('trigram_phraser.pkl')
   
    return (bigram_model, trigram_model)

def topic_word_weight_transformer(series):
    if series.name == 'word' or series.name == 'weight':
        return ','.join(series).upper()
    # elif series.name == 'weight':
    #     return np.array(series.values).astype('float')
    else:
        return series.values[0]
   
def doc_topic_mapping_transformer(series):
    if series.name == 'New_Keywords':
        return ','.join(series).upper()
    elif series.name == 'Keyword_Weights':
        # print(series)
        series = np.concatenate(list(series))
        return np.array2string(series, separator=',')
    else:
        return series.values[0]

if __name__ == '__main__':
    print('load document corpus')
    df = pd.read_pickle('../full_df.pickle')
   
    #remove duplicate CRs based on index
    df = df[df.OriginationDT >= min_origination_date]
    df = df[~df.index.duplicated(keep='first')]
   
    #ensure we have no nan entries
    df.Title[df.Title.isna()] = ''
    df.Condition[df.Condition.isna()] = ''
    df.ActionTaken[df.ActionTaken.isna()] = ''
    df.OperationalNotes[df.OperationalNotes.isna()] = ''
    df.IssueNotes[df.IssueNotes.isna()] = ''
    df.Recommendations[df.Recommendations.isna()] = ''
    df.EquipmentDescription[df.EquipmentDescription.isna()] = ''
   
    #clean up the data to remove numbers, punctuation, special characters and some odd html formatting
    df.Condition = df.Condition.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.Condition = df.Condition.apply(lambda x: x.strip())
    df.ActionTaken = df.ActionTaken.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.ActionTaken = df.ActionTaken.apply(lambda x: x.strip())
    df.OperationalNotes = df.OperationalNotes.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.OperationalNotes = df.OperationalNotes.apply(lambda x: x.strip())
    df.IssueNotes = df.IssueNotes.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.IssueNotes = df.IssueNotes.apply(lambda x: x.strip())
    df.Recommendations = df.Recommendations.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.Recommendations = df.Recommendations.apply(lambda x: x.strip())
    df.EquipmentDescription = df.EquipmentDescription.apply(lambda x: re.sub(r'(\\|\/|\(|\)|\<|\>|-|\*|#|[0-9])+', ' ', x.replace('&lt;', '<').replace('&amp;', 'and').replace('&quot;', '"').replace('div>', ' ').replace('br/>', ' ')))
    df.EquipmentDescription = df.EquipmentDescription.apply(lambda x: x.strip())
   
    #The condition field is probably the most important one for identifying the proposed keywords.
    #There are several CRs with very short and nondescript entries in the condition field.  These
    #can cause problems when generating the set of top-words for a CR, as these very short entries
    #do not provide meaningful results when processed using TF-IDF.  These short-entry CRs are removed.
    df = df[~((df.Condition.str.len() < 40) &
              (df.Condition.str.contains('cancel', flags=re.IGNORECASE) |
              df.Condition.str.contains('delete', flags=re.IGNORECASE) |
              df.Condition.str.contains('error', flags=re.IGNORECASE)))]
    df = df[df.Condition.str.len() > 40]

    corpus = pd.DataFrame(index = df.IssueID)
    corpus['text'] = df[text_fields].agg('. '.join, axis=1).values
    texts = corpus['text'].values
   
    # vec = CountVectorizer(stop_words=STOPWORDS)
    # texts_counted = vec.fit_transform(texts)
    # sum_words = texts_counted.sum(axis=0)
    # words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    # words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
   
    # text.lower().replace('reviewed', ' ').replace('issue', ' ').replace('required', ' ') for text in texts
   
    bow_corpus_and_dictionary = Path(bow_dictionary_file)
   
    # processed = list(map(preprocess, corpus.iloc[:100]['text']))
    # docs = generateNGrams(processed)
    # dictionary = gensim.corpora.Dictionary(docs)
    # dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=100000)
    # bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
   
    if not bow_corpus_and_dictionary.is_file():
        print('preprocess corpus')
        with Pool(processes=None) as pool: #None -> spawn as many processes as there are available CPUs
            corpus_batches = np.array_split(texts, cpu_count())
            processed = pool.map(preprocess, corpus_batches)
            processed = list(itertools.chain(*processed))
            print('number of documents preprocessed: {}'.format(len(processed)))
            print('done preprocessing corpus... Start Generating NGrams')
           
            docs = generateNGrams(processed)
            # ngram_batches = np.array_split(processed, cpu_count())
            # docs = pool.map(generateNGrams, ngram_batches)
            # docs = list(itertools.chain(*docs))
            print('done generating NGrams... Build BoW Corpus')
           
            dictionary = gensim.corpora.Dictionary(docs)
            dictionary.filter_extremes(no_below=3, no_above=0.2, keep_n=100000)
       
            bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
            print('done building BoW corpus... Start Mallet Model Generation')
           
        with open(bow_dictionary_file, 'wb') as f:
            pickle.dump([bow_corpus, dictionary, docs], f)
    else:
        print('load preprocessed training corpus')
        with open(bow_dictionary_file, 'rb') as f:
            bow_corpus, dictionary, docs = pickle.load(f)
       
    dictionary.id2token = dict((v,k) for k,v in dictionary.token2id.items())
    words_freq = [(dictionary.id2token[id], cnt) for id, cnt in dictionary.dfs.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_freq = pd.DataFrame(words_freq, columns=['word', 'count'])
    acronyms = words_freq[words_freq.word.str.len() <=3]
    acronyms.to_csv('acronyms.csv')
       
    if train:
        print('begin training mallet LDA model')
        mallet_lda_model = LdaMallet(path_to_mallet_binary,
                                     corpus=bow_corpus,
                                     iterations=3900,
                                     num_topics=140,
                                     alpha=60,
                                     id2word=dictionary,
                                     prefix=path_to_mallet_output,
                                     workers=multiprocessing.cpu_count())
        mallet_lda_model.save('{}lda_model.pkl'.format(path_to_mallet_output))
        # mallet_lda_model.save('{}lda_model_{}.pkl'.format(path_to_mallet_output, uuid))
        print('calculate model coherence C_v score')
        coherence_model_lda = CoherenceModel(model=mallet_lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('model coherence score: {}'.format(coherence_lda))
    else:
        print('load mallet LDA model')
        # mallet_lda_model = LdaMallet.load('{}lda_model.pkl'.format(path_to_mallet_output))
        mallet_lda_model = LdaMallet.load('{}lda_model_{}.pkl'.format(path_to_mallet_output, uuid))
   
    # # #convert the model to gensim format
    # # lda_model = malletmodel2ldamodel(mallet_lda_model)
    # # lda_model.save('{}gensim_lda_model.pkl'.format(path_to_mallet_output))
   
    # # topics = np.array(list(zip(*lda_model.show_topics(lda_model.num_topics, num_words=20, formatted=False)))[1])
    # # topics = topics[:, :, 0]
    # # topics = pd.DataFrame(topics)
    # # topics.to_csv('{}topics_with_20_words.csv'.format(path_to_mallet_output))
   
    # #determine which words will be included in each topic -> keyword string mapping
    # print('Determining word weight per topic')
    # topic_ids = mallet_lda_model.word_topics.argsort()[:, ::-1]
    # topic_weights = mallet_lda_model.word_topics
    # topic_weights.sort()
    # topic_weights = topic_weights[:, ::-1]
    # topic_weights = (topic_weights.T / topic_weights.max(axis=1)).T
    # valid_topic_word_idx = np.array(np.where(topic_weights > 0.1)).T
    # vocabulary = np.array(list(zip(*list(mallet_lda_model.id2word.id2token.items())))[1])
    # topic_words = vocabulary[topic_ids]
    # words_with_weights = np.dstack((topic_words, topic_weights))
   
    # topic_words = pd.DataFrame([(idx[0], words_with_weights[idx[0], idx[1]][0], words_with_weights[idx[0], idx[1]][1]) for idx in valid_topic_word_idx], columns=['topic_id', 'word', 'weight'])
    # topic_words['id_retain'] = topic_words.topic_id
    # topic_words = topic_words.groupby('topic_id').transform(topic_word_weight_transformer).drop_duplicates()
    # topic_words_weights = topic_words[['word','weight']].values
    # # # generic_words = np.array(['DESCRIPTION', 'SUBJECT','RECORDS','START','CONDITION','INDICATION','SAMPLE','CORRECT','EVALUATE','CHANGE','PROCEDURE','MEETING','REPLACEMENT','UNIT','ZONE','TIME','OPEN','BUILDING','EQUIPMENT','AREA'])
   
    # # # np.intersect1d(topic_words, generic_words)
   
    # #obtain the topic mapping for each document
    # print('Determining document weight per topic')
    # corpus_topic_weights = mallet_lda_model[bow_corpus]
    # corpus_topic_weights = np.array(corpus_topic_weights)
    # corpus_topic_weights = corpus_topic_weights[:, :, 1]
    # corpus_topic_ids = np.argsort(corpus_topic_weights)[:, ::-1]
   
    # #use the topic mapping to add a new set of keywords (the LDA topics) for those documents that qualify
    # print('Assigning new keywords for each document')
    # min_topic_probability = corpus_topic_weights.mean() + 5 * corpus_topic_weights.std()
    # valid_topic_docs = np.array(np.where(corpus_topic_weights > min_topic_probability)).T
    # corpus_topic_weights = corpus_topic_weights[valid_topic_docs[:, 0], valid_topic_docs[:, 1]]
    # doc_topic_mapping = pd.DataFrame(list(zip(valid_topic_docs[:, 0], #doc ids
    #                                           corpus_topic_weights[:], #doc topic weights
    #                                           topic_words_weights[valid_topic_docs[:, 1], 0],  #doc keywords
    #                                           topic_words_weights[valid_topic_docs[:, 1], 1])), #keyword weights
    #                                  columns=['id', 'Topic_Weights', 'New_Keywords', 'Keyword_Weights'])
    # doc_topic_mapping['Keyword_Weights'] = doc_topic_mapping['Keyword_Weights'].str.split(pat=',').map(lambda x: np.array(x).astype('float'))
    # doc_topic_mapping['Keyword_Weights'] = doc_topic_mapping['Topic_Weights'] * doc_topic_mapping['Keyword_Weights']
    # doc_topic_mapping['id_retain'] = doc_topic_mapping.id
    # doc_topic_mapping = doc_topic_mapping.groupby('id', as_index=False).transform(doc_topic_mapping_transformer).drop_duplicates()
    # doc_topic_mapping = doc_topic_mapping.set_index('id_retain', drop=True)
   
    # doc_topic_mapping['Keyword_Weights'] = doc_topic_mapping['Keyword_Weights'].map(lambda x: np.fromstring(x[1:-1], dtype=float, sep=','))
    # doc_topic_mapping['Keyword_Order'] = doc_topic_mapping['Keyword_Weights'].map(lambda x: np.argsort(x)[::-1])
    # doc_topic_mapping['New_Keywords'] = doc_topic_mapping['New_Keywords'].map(lambda x: np.array(x.split(',')))
   
    # doc_topic_mapping['New_Keywords'] = doc_topic_mapping[['New_Keywords', 'Keyword_Order']].apply(lambda x: x[0][x[1]][:8], axis=1)
    # doc_topic_mapping['Keyword_Weights'] = doc_topic_mapping[['Keyword_Weights', 'Keyword_Order']].apply(lambda x: x[0][x[1]][:8], axis=1)
    # doc_topic_mapping = doc_topic_mapping.drop(columns=['Topic_Weights','Keyword_Order'])
   
    # doc_topic_mapping[['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8']] = pd.DataFrame(doc_topic_mapping['New_Keywords'].tolist(), index = doc_topic_mapping.index)
    # doc_topic_mapping[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8']] = pd.DataFrame(doc_topic_mapping['Keyword_Weights'].tolist(), index = doc_topic_mapping.index)
   
    # doc_topic_mapping['New_Keywords'] = doc_topic_mapping['New_Keywords'].map(lambda x: np.array2string(x, separator=',')[1:-1].replace("'", '').replace(' ', ''))
    # doc_topic_mapping['Keyword_Weights'] = doc_topic_mapping['Keyword_Weights'].map(lambda x: np.array2string(x, separator=',')[1:-1].replace(' ', ''))
   
    # doc_topic_mapping['IssueID'] = corpus.iloc[doc_topic_mapping.index].index
    # doc_topic_mapping = doc_topic_mapping.set_index('IssueID', drop=True)
   
    # df = df.join(doc_topic_mapping, on='IssueID', lsuffix='_left', rsuffix='_right')
    # df.New_Keywords[df.New_Keywords.isna()] = ''
    # df.Keyword_Weights[df.Keyword_Weights.isna()] = ''
   
    # df.to_pickle('{}/Xcel_data_with_new_keywords.pkl'.format(path_to_mallet_output))
    # check = df[['Condition','Keywords', 'k1', 'w1', 'k2', 'w2', 'k3', 'w3', 'k4', 'w4', 'k5', 'w5', 'k6', 'w6', 'k7', 'w7', 'k8', 'w8']]
    # check.k1[check.k1.isna()] = ''
    # check.k2[check.k2.isna()] = ''
    # check.k3[check.k3.isna()] = ''
    # check.k4[check.k4.isna()] = ''
    # check.k5[check.k5.isna()] = ''
    # check.k6[check.k6.isna()] = ''
    # check.k7[check.k7.isna()] = ''
    # check.k8[check.k8.isna()] = ''
    # check.w1[check.w1.isna()] = ''
    # check.w2[check.w2.isna()] = ''
    # check.w3[check.w3.isna()] = ''
    # check.w4[check.w4.isna()] = ''
    # check.w5[check.w5.isna()] = ''
    # check.w6[check.w6.isna()] = ''
    # check.w7[check.w7.isna()] = ''
    # check.w8[check.w8.isna()] = ''
    # check.to_csv('{}/Xcel_data_with_new_keywords.csv'.format(path_to_mallet_output))

