#!/usr/bin/env python
# coding: utf-8

# # Data Read

# In[4]:


import numpy as np 
import pandas as pd

DATASET_COLUMNS = ['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
TWITTER_DATA = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
TWITTER_DATA.tail()


# In[6]:


TWITTER_DATA.head()


# # Data Analysis

# In[7]:


print(f"the length of dataset if {len(TWITTER_DATA)}")


# In[8]:


# Print basic information about the TWITTER_DATA object, like its size, data type, or structure., NULL VALUES ATC
TWITTER_DATA.info()


# In[15]:


#  how many times each unique value appears in the data.
TWITTER_DATA['target'].value_counts()


# In[ ]:


# import matplotlib.pyplot as plt
# Count the occurrences of each unique value
color_counts =TWITTER_DATA['target'].value_counts()

# Create a bar chart
plt.bar(color_counts.index, color_counts.values)
plt.xlabel('Color')
plt.ylabel('Count')
plt.title('Frequency of Colors in the Data')
plt.show()


# # Data Preprocessing

# In[19]:


DATA = TWITTER_DATA[['text','target']]


# In[20]:


DATA


# In[22]:


DATA['target'] = DATA['target'].replace(4,1)


# In[23]:


DATA


# In[29]:


DATA_POS = DATA[DATA['target'] == 1]
DATA_NEG = DATA[DATA['target'] == 0] #Separating positive and negative tweets
DATASET = pd.concat([DATA_POS, DATA_NEG])
DATASET.shape # 80000 each +ve & -ve sentiment


# In[30]:


DATASET


# In[32]:


DATASET['text'] = DATASET['text'].str.lower()


# In[33]:


DATASET


# In[35]:


import nltk

# Download the stopwords for English (or any other language supported by NLTK)
nltk.download('stopwords')

# Access the stopwords list
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS


# REMOVING STOPWORDS

# In[36]:


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
DATASET['text'] = DATASET['text'].apply(lambda text: cleaning_stopwords(text))
DATASET['text'].head()


#  CLEANING AND REMOVING PUNCTUATION

# In[37]:


import string
PUNCTUATIONS = string.punctuation
def cleaning_punctuations(text):
    translator = str.maketrans('', '', PUNCTUATIONS)
    return text.translate(translator)
DATASET['text'] = DATASET['text'].apply(lambda x: cleaning_punctuations(x))
DATASET['text'].tail()


# CLEANING AND REMOVING REPEATING CHARACTER

# In[38]:


import re
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
DATASET['text'] = DATASET['text'].apply(lambda x: cleaning_repeating_char(x))
DATASET['text'].tail()


# CLEANING AND REMOVING URL'S

# In[39]:


def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
DATASET['text'] = DATASET['text'].apply(lambda x: cleaning_URLs(x))
DATASET['text'].tail()


# CLEANING AND REMOVING NUMRIC NUMBERS

# In[40]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
DATASET['text'] = DATASET['text'].apply(lambda x: cleaning_numbers(x))
DATASET['text'].tail()


# GETTING TOKENIZATION OF TWEET TEXT

# In[ ]:


# !pip install keras_preprocessing


# In[43]:


from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('\s+', gaps = True)
DATASET['text'] = DATASET['text'].apply(tokenizer.tokenize)


# In[44]:


DATASET['text'].head()


# In[45]:


DATASET['text'].tail()


# APPLYING STEMMING

# In[46]:


'''stemming simply removes suffixes and prefixes based on rules, 
Words:

playing
am
better
Stemming:

playing -> play
am -> am
better -> better (conflation with "good")
Lemmatization:

playing -> play
am -> be (lemma for the verb "to be")
better -> good (lemma for the adjective "good")

while lemmatization considers the context and grammar to find the correct base form.'''


from functools import lru_cache
st = nltk.PorterStemmer()
stem = lru_cache(maxsize=50000)(st.stem)
def stemming_on_text(data):
    text = [stem(word) for word in data]
    return data
DATASET['text'] = DATASET['text'].apply(lambda x: stemming_on_text(x))
DATASET['text'].head()


# APPLYING LEMMATIZER

# In[60]:


nltk.download('wordnet')
get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora')


# In[62]:


nltk.download('omw-1.4')


# In[65]:


lm = nltk.WordNetLemmatizer()
lemmatize = lru_cache(maxsize=50000)(lm.lemmatize)
def lemmatizer_on_text(data):
    text = [lemmatize(word) for word in data]
    return data
DATASET['text'] = DATASET['text'].apply(lambda x: lemmatizer_on_text(x))
DATASET['text'].head(3)


# In[66]:


DATASET['text'].tail()


# In[67]:


DATASET


# # Plot a cloud of words for negative tweets

# In[71]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
data_neg = DATASET[DATASET['target']==0]['text'].apply(lambda x: ' '.join(x) )
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)
plt.axis('off')


# # Plot a cloud of words for positive tweets

# In[72]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
data_neg = DATASET[DATASET['target']==1]['text'].apply(lambda x: ' '.join(x) )
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)
plt.axis('off')


# Separating input feature and label

# In[74]:


words = set()
for data in DATASET['text']:
    for word in data:
        words.add(word)
        
print(f"the total word of all text is  {len(words)}")


# In[75]:


NEW_DATA = DATASET
NEW_DATA['text'] = DATASET['text'].apply(lambda x: ' '.join(x) )
NEW_DATA.head()


# In[76]:


max_features = len(words)
tokenizer_keras = Tokenizer(num_words=max_features, split=' ')
tokenizer_keras.fit_on_texts(NEW_DATA['text'].values)
X = tokenizer_keras.texts_to_sequences(NEW_DATA['text'].values)
X = pad_sequences(X)
y = pd.get_dummies(NEW_DATA['target']).values


# Splitting our data into Train and Test Subset

# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,
                                                    random_state =0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[78]:


validation_size = 240000

X_validate = X_test[-validation_size:]
y_validate = y_test[-validation_size:]
X_test = X_test[:-validation_size]
y_test = y_test[:-validation_size]


# LSTM Classifier

# In[80]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import Adam
from keras.regularizers import L2

embed_dim = 128
lstm_out = 196

# No TPU setup needed

# Define the model without the TPU strategy scope
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax', kernel_regularizer=L2(0.001)))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

print(model.summary())


# In[ ]:


from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)
history = model.fit(X_train, y_train, epochs = 20, batch_size=128, verbose = 2, validation_data=(X_validate, y_validate),
                    callbacks=[earlystopping]) 


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
score,acc = model.evaluate(X_test, y_test, verbose = 2, batch_size = 64)
print("score: %.2f" % (score))
print("accuracy : %.2f" % (acc))
y_pred = model.predict(X_test)
cfm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1), normalize='pred')
print(cfm)


# In[ ]:


sns.heatmap(cfm,annot=True,fmt='',linewidths=0.5)


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:




