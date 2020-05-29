from course_work.models import logistic_regression
from course_work.models import gaussian_NB
from course_work.models import multinominal_NB
from course_work.models import bernulli_NB
from course_work.models import vectorizer
import pandas as pd
import os


def predict(model, vectorizer, data):
    vec = vectorizer.transform(data)
    X = vec.toarray()
    y_pred = model.predict(X)
    predicted_data = pd.DataFrame({'Text': list(data), 'y_pred': y_pred})

    return predicted_data


if __name__ == "__main__":

    # data = YOUR_DATA

    model = logistic_regression()
    # model = gaussian_NB()
    # model = multinominal_NB()
    # model = bernulli_NB()

    predicted_data = predict(model, vectorizer, data)

