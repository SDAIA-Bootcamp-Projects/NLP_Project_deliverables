#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: noura
"""

import streamlit as st
import streamlit.components.v1 as stc
import sys
import spacy
import nltk
import pandas as pd
from itertools import combinations
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet
from nltk.corpus import words
import re
import itertools
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
from nltk import pos_tag
from wordcloud import WordCloud
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


df=pd.read_csv('/Users/noura/Desktop/SDAIA/NLP_Project_deliverables/df-for-app.csv')
df=df.drop(['Unnamed: 0'],axis=1)

cv_tfidf=TfidfVectorizer()
corpus=df['combined'].tolist()
cv_tfidf=TfidfVectorizer(stop_words="english")
X_tfidf=cv_tfidf.fit_transform(corpus).toarray()
dt_tfidf=pd.DataFrame(X_tfidf,columns=cv_tfidf.get_feature_names())

tfidf_matrix=cv_tfidf.fit_transform(df['combined'])
co_sim=linear_kernel(tfidf_matrix, tfidf_matrix)

indcs=pd.Series(df.index, index=df['title']).drop_duplicates()

def rec(title, cosine_sim=co_sim):
    idx=indcs[title]
    similarity=list(enumerate(cosine_sim[idx]))
    similarity=sorted(similarity, key=lambda x: x[1], reverse=True)
    similarity=similarity[1:11]
    idxs=[i[0] for i in similarity]
    return_items=df['title'].iloc[idxs]
    recommended_movies = []
    for t in return_items:
        recommended_movies.append(t)
    #print(recommended_movies)
        
    return recommended_movies


def main(): 
    THEMES = [
    "light",
    "dark",
    "green",
    "blue"]

    page_options = [' Recommender','About']

    
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == ' Recommender':
        # Header contents
        st.write('#  Recommendation Engine')
        # Recommender System algorithm selection

        
        #st.write('### Enter a Movie or Show to See Recommendations:')
        search_term=st.text_input('Enter the name of a  show or movie you like:')



        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = rec(search_term.lower())
                st.title("You might like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! I couldn't find that one. Try a different title.")





if __name__ == '__main__':
    main()


