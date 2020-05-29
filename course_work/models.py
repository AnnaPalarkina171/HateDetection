from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from nltk.corpus import stopwords
import pandas as pd
import os



stop_words = stopwords.words('russian')
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             analyzer='word',
                             lowercase=True,
                             ngram_range=(1, 1),
                             max_df=0.95,
                             min_df=4,
                             norm='l2',
                             use_idf=True)


def scores(model):
    print('Accuracy: ', round(model.score(X_test, y_test),4))
    # y_score = model.decision_function(X_test)
    # average_precision = average_precision_score(y_test, y_score)
    # print(f'Precision: {average_precision}')
    y_pred = model.predict(X_test)
    # recall = recall_score(y_test, y_pred)
    # print(f'Recall: {recall}')
    f1 = f1_score(y_test, y_pred)
    print(f'F1 score: {round(f1,4)}')


def logistic_regression():
    # для суржука
    # model = LogisticRegression(C=15.0, dual=False, penalty="l2")
    # для сексизма
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

def svc():
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    return model

def GBC():
    model = GradientBoostingClassifier(criterion='mse')
    model.fit(X_train, y_train)

    return model

def gaussian_NB():
    model = GaussianNB()
    model.fit(X_train, y_train)

    return model


def multinominal_NB():
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model


def bernulli_NB():
    model = BernoulliNB()
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    path = os.getcwd()
    parent = os.path.join(path, os.pardir)

    # for sexism
    # df = pd.read_csv(f'{os.path.abspath(parent)}\\data\\sexism\\stemmed_full_sexism.csv')

    # for ukraine hate
    df = pd.read_csv(f'{os.path.abspath(parent)}\\data\\ukraine_hate\\stemmed_full_ukraine.csv')

    # if we want model trained without preprocessing
    #vec = vectorizer.fit_transform(df['Text'])
    # if we want model trained with preprocessing
    vec = vectorizer.fit_transform(df['stemmed_text'].values.astype('U'))

    X = vec.toarray()
    y = list(df['Label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23434, stratify=y)

    print(scores(gaussian_NB()))
    print(scores(multinominal_NB()))
    print(scores(bernulli_NB ()))


    # log_reg = logistic_regression()
    # svc = svc()
    # gbc = GBC()
    # gaussian_nb = gaussian_NB()
    # multinominal_nb = multinominal_NB()
    # bernulli_nb = bernulli_NB()
    #
    # models = [log_reg,svc,gbc, gaussian_nb, multinominal_nb, bernulli_nb]
    # for model in models:
    #     scores(model)
    #     print('\n')
