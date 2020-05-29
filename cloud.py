from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os



stop_words = stopwords.words('russian')
vectorizer = CountVectorizer(stop_words=stop_words,
                             max_df=0.95,
                             min_df=4,
                             ngram_range=(1, 1),
                             analyzer='word',
                             )
#for sexism and hate corpora
#df = pd.read_csv('/data/sexism/full_corpora.csv', delimiter=',')
df = pd.read_csv('/data/ukraine_hate/full_corpora.csv', delimiter=',')

vec = vectorizer.fit_transform(df.Text.values)
features = vectorizer.get_feature_names()

X = vec.toarray()
y = list(df['Label'])

lreg_text = LogisticRegression(random_state=0)
lreg_text.fit(X, y)
coef = lreg_text.coef_[0]
intercept = lreg_text.intercept_

sent_words = pd.DataFrame({'feature': features,
                           'weight': [abs(c) for c in coef],
                           'value': ['hate' if c > 0 else 'non_hate' for c in coef]
                          }).sort_values(by='weight', ascending=False)

text = list(sent_words[sent_words['value'] == 'hate'].head(110)['feature'])
text = ' '.join(text)

comment_mask = np.array(Image.open("comment.jpg"))
cloud = WordCloud(background_color="white", max_words=100, mask=comment_mask)
cloud.generate(text)
plt.imshow(cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#for sexism and hate corpora
#cloud.to_file("post_cloud_sexism.png")
cloud.to_file("post_cloud_hate.png")
