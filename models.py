# models.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


def get_models():
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier())
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC(probability=True))
        ]),
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ]),
                'K-Nearest Neighbors': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', KNeighborsClassifier())
        ]),
        'Gradient Boosting': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', GradientBoostingClassifier())
        ])
    }
    return models