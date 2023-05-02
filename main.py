import pandas as pd
import numpy as np
import pickle
import  helper
import streamlit as st

model=pd.read_pickle("LogisticTFIDF.pkl")
vectorizer=pd.read_pickle("tfidfveactorizer.pkl")

st.title("Toxic Comment classification")

input=st.text_input("Enter a text")
clean_text=helper.complete_preprocessor(input)

text=vectorizer.transform([clean_text])

result=model.predict(text)[0]

if st.button("Predict"):
    if result==1:
        st.title("This is a toxic comment ")
    else:
        st.title("This is  not a toxic comment")