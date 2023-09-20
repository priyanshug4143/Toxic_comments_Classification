from flask import Flask , render_template , request
import pickle
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import logging

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = word_tokenize(text) 
    # remove any non ascii
    text = [word.encode('ascii', 'ignore').decode('ascii') for word in text]
    lmtzr = WordNetLemmatizer()
    text = [lmtzr.lemmatize(w) for w in text]
    text = [w for w in text if len(w) > 2]
    
    #Removing Stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    #Joining words to a single string
    cleaned_text = ' '.join(text)

    return cleaned_text 





# Configure the logging module
#log_file = "my_log_file.log"
#logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Log some messages

vector=pickle.load(open("Vectorizer.pkl" , "rb"))
predictor=pickle.load(open("svc.pkl" , "rb"))
#reprocessor=pickle.load(open("text__transformer.pkl" , "rb"))

app=Flask(__name__)

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/prediction" , methods=["POST"])
def predict_price():
    Texta1=request.form.get("Text")    
    Texta=clean_text(Texta1)
    vect=vector.transform([Texta])
    vect=list((vect.toarray()))
    vect=predictor.predict(vect)
    vect=list(vect.toarray())
    vect=list(vect[0])
    vect=sum(vect)
    if vect!=0:
        vect="This is a Toxic Comment"
    else :
        vect="This is Non Toxic Comment"

    return render_template("index.html" , final=vect , Texta=Texta1)


if __name__=="__main__":
    app.run(debug=True)
