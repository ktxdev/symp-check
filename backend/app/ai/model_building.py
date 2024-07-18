import nltk
import string
import joblib
import pandas as pd

from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


models = {
    # 'Logistic Regression': (LogisticRegression(multi_class='auto'), {
    #     'C': [0.1, 1, 10],
    #     'solver': ['liblinear']
    # }),
    # 'Support Vector Machine': (SVC(), {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'rbf']
    # }),
    # 'Random Forest': (RandomForestClassifier(), {
    #     'n_estimators': [10, 100, 200],
    #     'max_depth': [None, 10, 20]
    # }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }),
}

# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stop words
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    
    return text

def train_models(X, y):

    sss = StratifiedKFold(n_splits=2)

    # Test each model
    best_models = {}
    for model_name, (model, params) in models.items():
        grid_search = GridSearchCV(model, params, cv=sss, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)

        # best_model = grid_search.best_estimator_
        # y_pred = best_model.predict(X_test)

        # accuracy = accuracy_score(y_test, y_pred)
        # roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) if hasattr(best_model, "predict_proba") else None

        best_models[model_name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            # 'accuracy': accuracy,
            # 'roc_auc': roc_auc,
            # 'classification_report': classification_report(y_test, y_pred)
        }
    
    return best_models

if __name__ == "__main__":
    # Load the dataset
    ds = load_dataset("fhai50032/SymptomsDisease246k")

    # Convert to pandas DataFrame
    df = pd.DataFrame(ds['train'])

    # Preprocess the query text
    df['query'] = df['query'].apply(preprocess_text)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['query'])

    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['response'])


    # Training the models
    best_models = train_models(X, y)

    # Print results
    for model_name, result in best_models.items():
        print(f"Model: {model_name}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best Score: {result['best_score']}")
        # if result['roc_auc'] is not None:
        #     print(f"ROC-AUC: {result['roc_auc']}")
        # print(f"Classification Report:\n{result['classification_report']}")
        # print("\n" + "="*80 + "\n")

        # Save the model, vectorizer, and label encoder
        joblib.dump(result['best_model'], f'data/models/{model_name.replace(" ", "_")}_model.pkl')

    joblib.dump(vectorizer, 'data/models/tfidf_vectorizer.pkl')
    joblib.dump(label_encoder, 'data/models/label_encoder.pkl')
