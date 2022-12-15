#!/usr/bin/env python
# coding: utf-8

# # Flipkart Reviews Sentiment Analysis using Python

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data = pd.read_csv("flipkart_reviews.csv")
data.head()


# In[2]:


data.info()


# In[3]:


print(data.isnull().sum())


# In[4]:


import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[5]:


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[6]:


data["Review"] = data["Review"].apply(clean)
data.head()


# In[7]:


#The rating coloumn of the data contains the rating given by every reviewer.
#Here's the look of how most of the people rate the products they buy from flipkart.
ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=numbers,hole = 0.5)
figure.show()


# In[8]:


#Let's have a look at the kind of reviews people leave
text = " ".join(i for i in data.Review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(20,15))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[9]:


#Analysing the sentiments of Flipkart review by adding three coloumns in the dataset
#Positive, Negative and Neutral
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
data = data[["Review", "Positive", "Negative", "Neutral"]]
print(data.head())


# In[10]:


#Now we can see how most of the reviewers think about the products and services of Flipkart
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive 😊 ")
    elif (b>a) and (b>c):
        print("Negative 🙁 ")
    else:
        print("Neutral 😐 ")
sentiment_score(x, y, z)


# In[11]:


#Now we can see total of positive, negative and neutral sentiment scores to find a conclusion about flipkart reviews

print("Positive: ", x)

print("Negative: ", y)

print("Neutral: ", z)

