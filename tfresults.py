import os
import nltk
import re
import string
import copy
import unicodedata
from bs4 import BeautifulSoup
from os import listdir
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd

file_name = []
corpus = {}
queries = []
folder_dir = os.path.dirname(__file__)
rel_path = "../clueweb09PoolFilesTest/"
topic_path = "../pytf-idf/"
stop_words = set(stopwords.words('english'))
folderpath = os.path.join(folder_dir,rel_path)
counter = 1
tpath = os.path.join(folder_dir,topic_path)

#Gather file names
for file_ in os.listdir(folderpath):
    if("clueweb09-" in file_):
        file_name.append(file_)

#Sort query list
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#Create query list from 'topics.txt'
for subdir, dirs, files in os.walk(tpath):
    for file in files:
        file_path = subdir + os.path.sep + file
        if(file == "topics.txt"):
            with open(file_path, "r") as t:
                queries = [topic.strip() for topic in t]
                queries.sort(key=natural_keys)
            t.close()


#Stem words
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#Tokenize sentences
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    words = [i for i in tokens if not i in stop_words] #remove stop words
    stems = stem_tokens(words, stemmer)
    return stems

#Pre-processing step
print("Pre-processing documents...")
for file in file_name:
    if(counter < 201):
        start = timer()
        with open(os.path.join(folder_dir,rel_path,file), 'rb') as f:
            read_data = f.read() #Read from file

        input_str = BeautifulSoup(read_data, "lxml").get_text() # Extract text from document
        input_str = input_str.casefold() #Convert to lower-case
        input_str = re.sub(r'\d+', '', input_str) #Remove numbers 
        input_str = input_str.translate(str.maketrans("","",string.punctuation)) #Remove punctuation
        input_str = " ".join(input_str.split()) #Removes whitespaces
        input_str = input_str.replace("\n"," ") #Removes newline
        input_str = unicodedata.normalize("NFKD", input_str) #Removes unicode characters.
        corpus[file] = input_str
        print(counter)
        counter+=1
        f.close()
    else:
        break   
#print(list(corpus.values())[0]) --Print first document's text for testing
values = []
files = []

for k,v in corpus.items():
    values.append(v)
    files.append(k)

#TF-IDF Vectorization --Where it really slows down
print("Calculating TF-IDF on corpus...")
vectorizer = TfidfVectorizer(tokenizer = tokenize, sublinear_tf = True, use_idf=True, smooth_idf=True, max_df = 0.85)
tfidf = vectorizer.fit_transform(values)

#Search topics in corpus
print("Searching topics...")
for topic in queries:
    #print(topic)
    rank = 0
    with open("tfidf_result.test", "a+") as o:
        querytfidf = vectorizer.transform([topic])
        cs = linear_kernel(querytfidf,tfidf).flatten() #Cosine Similarity calculation
        related_doc_ids = [i for i in cs.argsort()[::-1]]
        for index in related_doc_ids:
            rank+=1
            score = cs[index]
            if(score > 0.0):
                o.write("{} Q0 {} {} {} Default\n".format(queries.index(topic)+1,files[index],rank,score))
    o.close()
    #print("\n") 
end = timer()
print("Done! Time elapsed: {} seconds".format(end-start))