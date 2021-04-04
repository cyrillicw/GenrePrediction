import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import spacy
from tqdm import tqdm
nlp = spacy.load('en_core_web_sm')

def normalize(s):
    doc = nlp(s)
    lemma_list = [token.lemma_ for token in doc]
    filtered_sentence = list(filter(lambda w: not nlp.vocab[w].is_stop, lemma_list))
    return ' '.join(filtered_sentence)

data = pd.read_csv('data_train.csv')
if 'text' not in data.columns:
    tqdm.pandas()
    data['text'] = data['plot_synopsis'].progress_apply(normalize)
    data.to_csv('data_train.csv',columns=data.columns, index=False)
mb = MultiLabelBinarizer()
data['splitted_tags'] = data['tags'].apply(lambda x: x.split(', '))
encoded_tags = mb.fit_transform(data['splitted_tags'])
total_score_train = 0
total_score_valid = 0
for i in range(encoded_tags.shape[1]):
    print(i)
    X_train_text, X_valid_text, y_train, y_valid\
        = train_test_split(data['text'], encoded_tags[:, i], test_size=0.3, random_state=9,
                           stratify=encoded_tags[:, i])
    pipe = Pipeline([
        ('vectorizer', TfidfVectorizer(max_df=0.8, max_features=10000)),
        ('clf', LogisticRegression(C=0.0001, class_weight='balanced'))
    ])
    pipe.fit(X_train_text, y_train)
    pred_train = pipe.predict(X_train_text)
    pred_valid = pipe.predict(X_valid_text)
    score_train = f1_score(y_train, pred_train)
    score_valid = f1_score(y_valid, pred_valid)
    total_score_train += score_train
    total_score_valid += score_valid
    print('Train Score: {}'.format(score_train))
    print('Valid Score: {}'.format(score_train))
    print()
total_score_train /= len(mb.classes_)
total_score_valid /= len(mb.classes_)
print('Total Train Score: {}'.format(total_score_train))
print('Total Valid Score: {}'.format(total_score_valid))
