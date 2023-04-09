import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the CSV file containing the news articles
df = pd.read_csv('News1.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['lable'], test_size=0.4, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training and testing data using the vectorizer
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Create a Passive Aggressive Classifier model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict the labels for the testing data
y_pred = pac.predict(tfidf_test)

# Calculate the accuracy score and confusion matrix
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", score)
print("Confusion Matrix:\n", cm)

# Dummy news for testing
print("enter any fake or real news: \n")
dummy_news = [input()]

# Transform the dummy news using the vectorizer
tfidf_dummy = tfidf_vectorizer.transform(dummy_news)

# Predict the labels for the dummy news
dummy_pred = pac.predict(tfidf_dummy)

# Print the predicted labels for the dummy news
for i in range(len(dummy_news)):
   
    print("Predicted label:", dummy_pred[i])
    print()

