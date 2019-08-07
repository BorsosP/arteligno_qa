# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:08:14 2019

@author: cameg
"""

from flask import Flask, request, jsonify
import pandas as pd
import json
from pandas.io.json import json_normalize
import spacy
import random
from sklearn.externals import joblib
from sklearn_crfsuite import CRF


crf = joblib.load('answer_tagger_cnn_based_02')
nlp = spacy.load('en_core_web_sm')


def pos_tagger(sentence):
    doc = nlp(sentence)
    tags = []
    for token in doc:
        tags.append(token.tag_)
    return tags


def tokenizer(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        tokens.append(token.text)
        
    if tokens == []:
        return 'mv-9'
    else:
        return tokens
    
def entity_recognizer(sentence):
    doc = nlp(sentence)
    entities = []
    for token in doc:
        entities.append(token.ent_type_)

    if entities == []:
        return 'mv-9'
    else:
        return entities
    
    
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, e) for w, p, e in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["NER"].values.tolist())]
        self.grouped = self.data.groupby("sentence_num_Word").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
        
        
        
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    ner = sent[i][2]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'ner' : ner
    }
    if i == 1:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:ner': ner1
        })
        
    elif i > 1:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:ner': ner1
        }),
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner2 = sent[i-2][2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:ner': ner2
        })
        
        
    else:
        features['BOS'] = True

    if i < len(sent)-2:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:ner': ner1
        }),
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner2 = sent[i+2][2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:ner': ner2
        })
    elif i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:ner': ner1
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, ner, label in sent]

def sent2tokens(sent):
    return [token for token, postag, ner, label in sent]



def question_generator_base(df, loc):
    orig = df['Word'].iloc[loc].copy()
    quest = df['Word'].iloc[loc].copy()
    pred = df['pred_flag'].iloc[loc].copy()
    filtered_lst = [(x,y) for x,y in enumerate(pred) if y == 'ANSWER']
    
    index_needs = [elem[0] for elem in filtered_lst]
    
    for elem in index_needs:
        quest[elem] = '_______'
        
    return ' '.join(orig), ' '.join(quest)
  




app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    
    payload = request.json['data']
        
    sentences = []
    doc = nlp(payload)
    
    for sentence in doc.sents:
        sentences.append(sentence.text)
    
    sentence_df_unique = pd.DataFrame(sentences, columns=['sentence'])
    
    sentence_df_unique['sen_len'] = sentence_df_unique['sentence'].apply(lambda 
                      text: len(text.split()))
    sentence_df_unique['POS'] = sentence_df_unique['sentence'].apply(pos_tagger)
    sentence_df_unique['Word'] = sentence_df_unique['sentence'].apply(tokenizer)
    sentence_df_unique['NER'] = sentence_df_unique['sentence'].apply(entity_recognizer)
    
    sentence_df_unique['sentence_num'] = sentence_df_unique.index
    
    sentence_df_proc = pd.DataFrame()
    
    features = ['POS', 'Word', 'NER']
    
    for feature in features:
        train_data_need = sentence_df_unique[['sentence_num', feature]]
        transformed = (train_data_need[feature].apply(pd.Series)
                  .merge(train_data_need, right_index = True, left_index=True)
                  .drop([feature], axis=1)
                  .melt(id_vars = ['sentence_num'], value_name=feature)
                  .drop('variable', axis=1)
                  .dropna())
        transformed.rename(columns={'sentence_num' : 'sentence_num_' + feature}, inplace=True)
        sentence_df_proc = pd.concat([sentence_df_proc, transformed ], axis=1)
    
    sentence_df_proc.drop(['sentence_num_POS', 'sentence_num_NER'], axis=1, inplace=True)
        
    sentence_df_proc.reset_index(inplace=True)
    sentence_df_proc.sort_values(['sentence_num_Word', 'index'], inplace=True)
    
    getter = SentenceGetter(sentence_df_proc)
    sentences = getter.sentences
    
    X = [sent2features(s) for s in sentences]
    
    pred = crf.predict_marginals(X)
    
    pred = [item for sublist in pred for item in sublist]
    
    pred = pd.DataFrame(pred)
    
    pred['pred_flag'] = pred.apply(lambda df: 'ANSWER' if df['ANSWER'] > df['O'] else 'O', axis=1)
    
    sentence_df_proc = sentence_df_proc[['sentence_num_Word', 'Word']].copy()
    sentence_df_proc.reset_index(drop=True,inplace=True)

    sentence_df_proc_final = pd.concat([sentence_df_proc, pred], axis=1)

    sentence_df_proc_ans = (sentence_df_proc_final[['sentence_num_Word', 'pred_flag']]
                            .groupby('sentence_num_Word')['pred_flag'].apply(list))

    sentence_df_proc_sen_list = (sentence_df_proc_final[['sentence_num_Word', 'Word']]
                            .groupby('sentence_num_Word')['Word'].apply(list))

    sentence_df_proc_sen = pd.concat([ sentence_df_proc_ans, sentence_df_proc_sen_list], axis=1).reset_index()
    
    output = []
    for i in range(0, len(sentence_df_proc_sen)):
        orig, quest = question_generator_base(sentence_df_proc_sen, i)
        output.append([orig, quest])
        
    return jsonify({'prediction': str(output)})
        
    
    

if __name__ == '__main__':
    
    
    app.run(debug=True)

