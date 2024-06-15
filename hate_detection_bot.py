import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import nltk
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from vectorizers import *
import time

st.title('Hate Speech Detection chat')



def tfidf_vectorizer():
    vectorizer = joblib.load(open('pickled_file/final_tfidf.pkl','rb')) 
    return vectorizer
#print(vectorizer)

def pos_vectorizer():
    pos_vectorizer = joblib.load(open('pickled_file\\final_pos.pkl','rb'))
    return pos_vectorizer


def model():
    model = joblib.load(open('pickled_file\\final_model.pkl','rb'))
    return model

def model_preprocessing(input_str):
    test_tfidf_ = tfidf_vectorizer().fit_transform([input_str]).toarray()
    tweet_tags = []
    tokens = joblib.load(open('pickled_file\\final_tokens.pkl','rb'))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)
    pos_ = pos_vectorizer().fit_transform(tweet_tags).toarray()
    feats_ = get_feature_array_([input_str])
    fin = np.concatenate([test_tfidf_, pos_, feats_],axis=1)
    pred = model().predict(fin)
    return pred

def response_generator(prompt):
    pred = model_preprocessing(prompt)
    response = f"Model says: {'🟢 No issues with this text!' if pred[0] == 2 else '🚩 Hey! this is a hate speech' if pred[0] == 0 else '⚠️ Hey this is offensive language'}"
    for word in response.split():
        yield word + " "
        time.sleep(0.1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(f'User : {prompt}')
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        #pred = model_preprocessing(prompt)
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

