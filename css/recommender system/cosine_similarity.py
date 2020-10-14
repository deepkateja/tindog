from sklearn.feature_extraction.text import CountVectorizer

text=['london paris london','paris paris london']
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(text)

print(X.toarray())

from sklearn.metrics.pairwise import cosine_similarity
cs=cosine_similarity(X)
print(cs)
