from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)

model = MultinomialNB()
model.fit(X_train_tfidf, train_data.target)

X_test_counts = count_vect.transform(test_data.data)
X_test_tfidf = tfidf.transform(X_test_counts)
predicted = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(test_data.target, predicted))
print("\nClassification Report:\n", classification_report(test_data.target, predicted, target_names=test_data.target_names))
print("Confusion Matrix:\n", confusion_matrix(test_data.target, predicted))
