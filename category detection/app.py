import os
import numpy as np
import hazm 
from hazm import word_tokenize
import pickle
import random
import nltk
import json   
stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
import streamlit as st


current_dir = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(current_dir,'models','lion_v.jsdh')
lion_v = open(model,'rb')
vectorizer = pickle.load(lion_v)
lion_v.close()

model = os.path.join(current_dir,'models','lion_le.jsdh')
lion_le = open(model,'rb')
le = pickle.load(lion_le)
lion_le.close()

model= os.path.join(current_dir,'models','lion_svc.jsdh')
lion_svc = open(model,'rb')
svc = pickle.load(lion_svc)
lion_svc.close()

model = os.path.join(current_dir,'models','stopwords.txt')
with open(model) as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [lin.replace('\n','') for lin in stopwords]

st.title('News Category Detection with AI!')
text = st.text_area('Enter your text news : ')
btn = st.button('Detect Category')

if btn:
    if len(text) >=10:
        title_body_tokenized = word_tokenize(text)
        title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
        title_body_tokenized_filtered_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered]
        title_body_tokenized_filtered_lemmatized = [lemmatizer.lemmatize(w).replace('#',' ') for w in title_body_tokenized_filtered]
        x=[' '.join(title_body_tokenized_filtered_lemmatized)+' '+ ' '.join(title_body_tokenized_filtered_stemmed)]
        x_v= vectorizer.transform(x)
        p = svc.predict(x_v)
        Label = le.inverse_transform(p)
        st.success('عنوان خبر شما :‌' + str(Label[0]))
    else:
        st.error('متن خبر شما باید از 10 کلمه بیشتر باشد')
    