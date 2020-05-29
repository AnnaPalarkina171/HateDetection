import pandas as pd
import regex as re
import nltk
import os
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords

russian_stemmer = nltk.stem.snowball.SnowballStemmer("russian")
lemmatizer = Mystem()
russian_stopwords = stopwords.words('russian')


def remove_with_regex(text):
    regex = re.compile("\[[i][d][0-9]*\|.*],")
    regex1 = re.compile('.*писал\(а\):')
    regexp = re.compile('[\r\n]')
    words = [re.sub(regex, "", word) and re.sub(regex1, "", word) and re.sub(regexp, "", word) for word in text.split()]
    return ' '.join(words)


def remove_stowords(text):
    text = [word for word in text.split() if word.lower() not in russian_stopwords]
    return ' '.join(text)


def remove_punctuation_and_make_lowercase(text):
    sentences = [(sentence.translate(str.maketrans('', '', punctuation))).lower() for sentence in text.split()]
    return ' '.join(sentences)


def lemmatization(text):
    text = lemmatizer.lemmatize(text)
    return ''.join(text)


def stemmatize(text):
    text = [russian_stemmer.stem(word) for word in text.split()]
    return ' '.join(text)


def corpora_with_stemmed_column(data):
    new_column = pd.DataFrame(columns=['stemmed_text'])
    for text in list(data['Text']):
        stemmed_text = stemmatize(
            lemmatization(remove_punctuation_and_make_lowercase(remove_stowords(remove_with_regex(text)))))
        df = pd.DataFrame({'stemmed_text': stemmed_text}, index=[0])
        new_column = new_column.append(df, ignore_index=True)
    data['stemmed_text'] = list(new_column['stemmed_text'])

    return data


if __name__ == "__main__":
    path = os.getcwd()
    parent = os.path.join(path, os.pardir)
    # for sexism and hate corpora
    data = pd.read_csv(f'{os.path.abspath(parent)}\\data\\sexism\\full_corpora.csv')
    # data = pd.read_csv(f'{os.path.abspath(parent)}\\data\\ukraine_hate\\full_corpora.csv')

    new_corpora = corpora_with_stemmed_column(data)

    # for sexism and hate corpora
    new_corpora.to_csv(f'{os.path.abspath(parent)}\\data\\sexism\\stemmed_full_sexism.csv',
                columns=['Label', 'Text', 'stemmed_text'], index=False)
    # new_corpora.to_csv(f'{os.path.abspath(parent)}\\data\\ukraine_hate\\stemmed_full_ukraine.csv',
    #                    columns=['Label', 'Text', 'stemmed_text'], index=False)

