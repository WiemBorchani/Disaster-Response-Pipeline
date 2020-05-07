import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
       Function:
       load data from database

       Args:
       database_filepath: the path of the database

       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
       """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Disaster-Response', engine)
    X = df['message']  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    return X, Y


def tokenize(text):
     """
    Function: processing the message 
    Args:
      text(str): the message
    Return:
      lemmed: a list of the root form of the message words
    """
   # Normalize
     text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
   # Tokenize 
     words = word_tokenize(text)
   # Remove stop_words
     words = [w for w in words if w not in stopwords.words("english")]
   # Lemmatize
     lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
     return lemmed


def build_model():
    """
     Function: build a model for classifing the disaster messages

     Return:
       cv(list of str): classification model
     """

    # Pipleine1 : grid Random Forest Classifier
    pipeline1 = Pipeline([
      ('vect', CountVectorizer(tokenizer = tokenize)),
      ('tfidf', TfidfTransformer()),
      ('clf',  MultiOutputClassifier(RandomForestClassifier()))
                   ])
    # Create Grid search parameters
   # Pipleine 2: grid search Classifier 
    parameters =  {
      'tfidf__use_idf': (True, False),
       'clf__estimator__n_estimators': [50, 60, 70]
            }

    cv = GridSearchCV(pipeline1, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
    Args:
    model: the classification model
    X_test: test messages
    Y_test: test target
    """
    y_pred = model.predict(X_test)
    i = 0
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
        accuracy = (y_pred == y_test).mean()
        print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the model
    Args:
    model: the classification model
    model_filepath (str): the path of pickle file
    """

   

     # Create a pickle file for the model
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()