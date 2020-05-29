import pandas as pd

links = [
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/ant_1.csv',
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/ant_2.csv',
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/media_1.csv',
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/media_2.csv',
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/media_3.csv']

data = pd.read_csv(
    'https://raw.githubusercontent.com/clips/gsoc2019_crosslang/master/russian%20sexist%20corpora/annotated/ant_1.csv',
    delimiter=',')
sexist_corpora = data[data['Label'] == 'sexist'][['Label', 'Text']]
non_sexist_corpora = data[data['Label'] == 'non_sexist'][['Label', 'Text']]

for link in links:
    new_data = pd.read_csv(link, delimiter=',')
    new_sexist_corpora = new_data[new_data['Label'] == 'sexist'][['Label', 'Text']]
    new_non_sexist_corpora = new_data[new_data['Label'] == 'non_sexist'][['Label', 'Text']]
    sexist_corpora = sexist_corpora.append(new_sexist_corpora, ignore_index=True)
    non_sexist_corpora = non_sexist_corpora.append(new_non_sexist_corpora, ignore_index=True)

non_sexist_corpora = non_sexist_corpora.head(4000)

sexist_corpora.to_csv('sexist_corpora.csv', index=False)
non_sexist_corpora.to_csv('non_sexist_corpora.csv', index=False)


df1 = pd.read_csv('non_sexist_corpora.csv', delimiter=',')
df2 = pd.read_csv('sexist_corpora.csv', delimiter=',')

data = df1.append(df2, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('full_corpora.csv', columns=['Label', 'Text'], index=False)




