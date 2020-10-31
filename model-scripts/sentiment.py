import pandas as pd
import numpy as np

df = pd.read_csv("/home/dhruv/Documents/python/ai-application/datasets/sentimentdata.csv")
x = np.array(df["review"])
y = np.array(df["sentiment"])
print("Data processed")

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 42)
print("Data split")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
print("Vectorizer initialized")

train_vectors = vectorizer.fit_transform(train_x)
test_vectors = vectorizer.transform(test_x)
print("Transform complete")


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(train_vectors, train_y)
print("Model trained")

import pickle

with open('/home/dhruv/Documents/python/ai-application/models/sentiment_classifier.pkl','wb') as f:
    pickle.dump(log, f)

with open('/home/dhruv/Documents/python/ai-application/models/sentiment_vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer, f)

print("DONE")




