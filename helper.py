import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pickle
ps=PorterStemmer()
def text_transformer(text):
    # 1 lower()
    text = text.lower()

    # 2 tokenizer
    text = nltk.word_tokenize(text)
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]

    y.clear()

    # 3 and 4 cheching stopwords and puctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    # 5 stemming
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def remove_hashtag(string):
    special_characters = re.compile(r'#+')
    return special_characters.sub(' ', string)

def remove_special_characters(string):
    special_characters = re.compile(r'[^A-Za-z0-9 \!\?#]+')
    return special_characters.sub('', string)

def remove_extra_whitespaces(string):
    extra_whitespaces = re.compile(r'\s+')
    return extra_whitespaces.sub(' ', string).strip()

def complete_preprocessor(text):
    text=text_transformer(text)
    text=remove_hashtag(text)
    text=remove_special_characters(text)
    text=remove_extra_whitespaces(text)
    return text