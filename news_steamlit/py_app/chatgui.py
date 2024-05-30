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

current_dir = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(current_dir,'models','lion_v.jsdh')
lion_v = open(model,'rb')
vectorizer = pickle.load(lion_v)
lion_v.close()

model = os.path.join(current_dir,'models','lion_le.jsdh')
lion_le = open(model,'rb')
le = pickle.load(lion_le)
lion_le.close()

model = os.path.join(current_dir,'models','lion_svc.jsdh')
lion_svc = open(model,'rb')
svc = pickle.load(lion_svc)
lion_svc.close()

model = os.path.join(current_dir,'models','stopwords.txt')
with open(model) as stopwords_file :
    stopwords = stopwords_file.readlines()
stopwords = [line.replace('\n','') for line in stopwords]

nltk_stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(nltk_stopwords)

def predict_class(news):
    title_body_tokenized = word_tokenize(news)
    title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
    title_body_tokenized_filtered_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered]
    title_body_tokenized_filtered_lemmatized = [lemmatizer.lemmatize(w).replace('#',' ') for w in title_body_tokenized_filtered]
    x=[' '.join(title_body_tokenized_filtered_lemmatized)+' '+ ' '.join(title_body_tokenized_filtered_stemmed)]
    x_v= vectorizer.transform(x)
    p = svc.predict(x_v)
    Label = le.inverse_transform(p)
    return Label[0]

def send():
    msg = Entry_box.get('1.0','end-1c').strip()
    Entry_box.delete('0.0',END)
    if msg!='':
        chatlog.config(state=NORMAL)
        ras = predict_class(msg)
        chatlog.insert(END,'Label is'+ras+'\n')
        chatlog.config(state=DISABLED)
        chatlog.yview(END)
        
def _onKeyRelease(event):
    cntrl = (event.state & 0*4)!=0
    if event.keycode==88 and cntrl and event.keysym.lower()!='x':
        event.widget.event_generate('<<Cut>>')
    if event.keycode==86 and cntrl and event.keysym.lower()!='v':
        event.widget.event_generate('<<Paste>>')
    if event.keycode==67 and cntrl and event.keysym.lower()!='c':
        event.widget.event_generate('<<Copy>>')
    
import tkinter
from tkinter import *

base = Tk()
base.title('Hello')
base.geometry('500x450')
base.resizable(width=True,height=True)

chatlog = Text(base,bd=0,bg='gray',height='8',width='50',font='Arial')
chatlog.config(state=DISABLED)

scrollbar = Scrollbar(base,command=chatlog.yview)
chatlog['yscrollcommand'] = scrollbar.set

sendButton = Button(base,font=('Verdana',12,'bold'),text='Send',height='5',width='12',bd=0,bg='#32de97',activebackground='#3c9d9b',fg='#fff',command=send)
Entry_box = Text(base,bd=0,bg='gray',height='5',width='29',font='Arial')
Entry_box.bind_all('<Key>',_onKeyRelease,'+')
scrollbar.place(x=376,y=6,height=386)
chatlog.place(x=6,y=6,height=386,width=370)
Entry_box.place(x=128,y=401,height=90,width=265)
sendButton.place(x=6,y=401,height=90)

base.mainloop()