from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import pandas as pd
import pickle
import numpy as np


def clear_punctuation(text):
    punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    for i in punctuation:
        text.replace(i, "")
    words = text.split()
    for i in range(len(words)):
        if words[i][0] == "@":
            words[i] = "@"
        words[i] = words[i].lower()
    return "".join(words)


def learn_model():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('./dataset/positive.csv', sep=';', on_bad_lines="skip", names=n, usecols=['text'])
    data_negative = pd.read_csv('./dataset/negative.csv', sep=';', on_bad_lines="skip", names=n, usecols=['text'])

    sample_size = 110_000
    reviews_without_shuffle = np.concatenate((data_positive['text'].values[:sample_size],
                                              data_negative['text'].values[:sample_size]), axis=0)
    labels_without_shuffle = np.asarray([1] * sample_size + [0] * sample_size)

    assert len(reviews_without_shuffle) == len(labels_without_shuffle)

    reviews, labels = shuffle(reviews_without_shuffle, labels_without_shuffle, random_state=0)

    x_train, y_train = reviews, labels
    s = 176_000
    x_train, y_train = x_train[:s], y_train[:s]
    x_train = [clear_punctuation(i) for i in x_train]
    vect = CountVectorizer(binary=True).fit(x_train)
    x_train = vect.transform(x_train)
    classifier = LogisticRegression(max_iter=25000)
    classifier.fit(x_train, y_train)
    return vect, classifier


def test():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('./dataset/positive.csv', sep=';', on_bad_lines="skip", names=n, usecols=['text'])
    data_negative = pd.read_csv('./dataset/negative.csv', sep=';', on_bad_lines="skip", names=n, usecols=['text'])

    sample_size = 110_000
    reviews_without_shuffle = np.concatenate((data_positive['text'].values[:sample_size],
                                              data_negative['text'].values[:sample_size]), axis=0)
    labels_without_shuffle = np.asarray([1] * sample_size + [0] * sample_size)

    assert len(reviews_without_shuffle) == len(labels_without_shuffle)

    reviews, labels = shuffle(reviews_without_shuffle, labels_without_shuffle, random_state=0)

    x_train, y_train = reviews, labels
    s = 176_000
    x_test, y_test = x_train[s:], y_train[s:]
    x_test = [clear_punctuation(i) for i in x_test]
    print(len(x_train), len(x_test))
    x_test = vect.transform(x_test)
    print(accuracy_score([classifier.predict(i) for i in x_test], y_test))
    print(recall_score([classifier.predict(i) for i in x_test], y_test))
    print(precision_score([classifier.predict(i) for i in x_test], y_test))
    print(f1_score([classifier.predict(i) for i in x_test], y_test))


def save_model(vect, classifier):
    with open('vect1.pickle', 'wb') as f:
        pickle.dump(vect, f)

    with open('classifier1.pickle', 'wb') as f:
        pickle.dump(classifier, f)


def load_model():
    with open('vect.pickle', 'rb') as f:
        vect = pickle.load(f)
    with open('classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    return vect, classifier


def predict(text):
    global classifier, vect
    return classifier.predict(vect.transform([text]))[0]


vect, classifier = load_model()
if __name__ == "__main__":
    TRAIN = False

    if TRAIN:
        vect, classifier = learn_model()
        save_model(vect, classifier)
    else:
        vect, classifier = load_model()
        test()
