# import libraries
import re
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sys


def load_data(database_filepath):
    """ Load data from the database
        engine: variable load the DRP.db database
        df: variable is a dataset that was created from databse
        split the dataset to X and Y
        X contains the message coloumn
        Y is the all coloumns except the message coloumn
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("DRP", engine)
    X = df["message"]
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns.values)
    return X, Y, category_names


def tokenize(text):
    """ Creat a tokenize function and apply it on the text
    url_regex: reqular expression for the urls in text
    deteced_urls:to find all urls
    tokens: split text to words
    lemmatizer: return  the word to it base from
    clean_tokens: the clean tokens from the text
    """
    #     import nltk
    # download the required nltk
    #     nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])

    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_.&+]|[!*\(\),]|(?:%[0-9FA-F][0-9a-fA-F][0-9a-fA-F]))+"
    deteced_urls = re.findall(url_regex, text)
    for url in deteced_urls:
        text = text.replace(url, "urlplaceholder")
    # split text to word
    tokens = word_tokenize(text)
    # back the word to it base from
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ("vect()", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)


def save_model(model, model_filepath):
    import pickle

    file = "classifier.pk1"
    pickle.dump(model, open(file, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
