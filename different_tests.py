import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('C:\\Users\\annet\\PycharmProjects\\HateDetection\\data\\sexism\\stemmed_full_sexism.csv')

stop_words = stopwords.words('russian')
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             analyzer='word',
                             lowercase=True,
                             ngram_range=(1, 1),
                             max_df=0.95,
                             min_df=4,
                             norm='l2',
                             use_idf=True)
vec = vectorizer.fit_transform(df['stemmed_text'].values.astype('U'))
X = vec.toarray()
y = list(df['Label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23434)


def scores(model):
    print('Accuracy on test: ', model.score(X_test, y_test))
    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print(f'Precision: {average_precision}')
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    print(f'Recall: {recall}')
    f1 = f1_score(y_test, y_pred)
    print(f'F1 score: {f1}')


def gbc():
    model = GradientBoostingClassifier(criterion='mse')
    model.fit(X_train, y_train)
    return model


def svc():
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


# print(scores(gbc()), '\n\n')
# print(scores(svc()), '\n\n')




from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def matrix(classifier):
#    classifier = GradientBoostingClassifier(criterion='mse').fit(X_train, y_train)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=['sexist','non sexist'], #NONE
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

# print(matrix(gbc()))
# print(matrix(svc()))







def visualize_coefficients(classifier, feature_names, n_top_features=25):
    # get coefficients with large absolute values
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha="right");
    plt.show()

def plot_grid_scores(grid, param_name):
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'],
    color='green', label='train')
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'],
    color='red', label='test')
    plt.legend();

# logit = log_reg()
# print(visualize_coefficients(logit, vectorizer.get_feature_names()))






# model = LogisticRegression(C=15.0, dual=False, penalty="l2")
# model.fit(X_train, y_train)
# print('Logistic regression',scores(model))

# model = SVC(kernel='linear')
# model.fit(X_train, y_train)
# print('SVC: ',scores(model))

# model = GradientBoostingClassifier(criterion='mse')
# model.fit(X_train, y_train)
# print('Gradient Boosting: ',scores(model))

# model = GaussianNB()
# model.fit(X_train, y_train)
# print(scores(model))

# model = MultinomialNB()
# model.fit(X_train, y_train)
# print(scores(model))

# model = BernoulliNB()
# model.fit(X_train, y_train)
# print(scores(model))






