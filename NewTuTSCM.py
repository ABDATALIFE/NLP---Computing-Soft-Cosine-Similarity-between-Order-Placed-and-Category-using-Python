#%% 1
from gensim import corpora 
import jieba
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.test.utils import common_texts
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
#from multiprocessing import cpu_count
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
import pandas as pd
import numpy as np

#%% 2
cart = pd.read_csv('gng_cart_sales.csv')
inv = pd.read_csv('gng_inventory.csv')
cart_KFC = cart[(cart.store_name=='KFC') | (cart.store_name=='KFC BLR')]


#%% 3 -- Cleaning the datasets libraries
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


#%% 4--
names = [i.lower() for i in cart_KFC['name']]
puncs = list("0123456789?:!.,&;()+") # Replacing punctuations  and symbols with nothing
for i in puncs:
    names = [sub.replace(i, '') for sub in names]

#%% 5--
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]    
    
    
    
for i in range(0,1000):
    corpus.append(preprocess(names[i]))



#%%
documents = []
from gensim.corpora import Dictionary
for i in range(0,1000):
    documents.append(corpus[i])
#%%
dictionary = Dictionary(documents)
#%%
newsentences =[]

for i in range(0,1000):
    newsentences.append(dictionary.doc2bow(corpus[i]))


#%%
newdocuments = []
from gensim.models import TfidfModel
for i in range(0,1000):
    newdocuments.append(newsentences[i])


#%%
tfidf = TfidfModel(newdocuments)


#%%

maindocuments = []
for i in range(0,1000):
    maindocuments.append(tfidf[newdocuments[i]])
    
    
    
#%%
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
termsim_index = WordEmbeddingSimilarityIndex(model)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)



#%%
similarity = termsim_matrix.inner_product(maindocuments[5],maindocuments[6], normalized=(True, True))
print('similarity = %.4f' % similarity)


#%%
len_array = np.arange(len(newsentences))
xx, yy = np.meshgrid(len_array, len_array)



#%%
column_names = names[0:1000]

df = pd.DataFrame(index = column_names,columns=column_names)
#%%
for i in range(0,1000):
    for j in range(0,1000):
        df= df.append()
        
#%%
cossim_mat = pd.DataFrame([[(termsim_matrix.inner_product(maindocuments[i],maindocuments[j], normalized=(True, True))) for i, j in zip(x,y)] for y, x in zip(xx, yy)],index=column_names,columns=column_names)


#%%
cossim_mat.to_csv('df_soft_cosine.csv')
