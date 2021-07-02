import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import gensim
from gensim.models import Word2Vec
import numpy as np
import pickle
from scipy.sparse import hstack

data = pd.read_csv("data.csv")
print("shape of the dataset:", data.shape)

stop_words = set(stopwords.words('english'))
splitted_sent_list = []
ps = PorterStemmer()

def text_preprocessing(total_text, index, column):
    '''This function cleans text, remove stopwords and symbols, lowercase the text and stem the text'''

    if type(total_text) is not int:
        total_text = re.sub(r'[^a-zA-Z0-9\n]', " ", total_text).lower()
        total_text = re.sub(r'\s+', " ", total_text)
        word_list = total_text.split()
        word_list = [ps.stem(word) for word in word_list]

        word_list = [word for word in word_list if word not in stop_words]
        word_list = [word for word in word_list if len(word) > 1]
        splitted_sent_list.append(word_list)




data.fillna('NA', inplace=True)

data.drop_duplicates(subset=['Title', 'Text'], inplace=True, ignore_index=True)
data['Cat1_Cat2_Cat3'] = data['Cat1'] + "$" + data['Cat2'] + "$" + data['Cat3']


start_time = time.clock()
for index, row in data.iterrows():
    if type(row['Title']) is str:
        text_preprocessing(row['Title'], index, 'Title')
    else:
        print("there is no text description for id:",index)
print('Time taken for preprocessing the text :',time.clock() - start_time, "seconds")




start_time = time.clock()
for index, row in data.iterrows():
    if type(row['Text']) is str:
        text_preprocessing(row['Text'], index, 'Text')
    else:
        print("there is no text description for id:",index)
print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")

merged_title_text = [" ".join(splitted_sent_list[i]) + ", index, column " + " ".join(splitted_sent_list[i+data.shape[0]]) for i in range(len(data))]
data['cleaned_title_text'] = merged_title_text

def get_input_matrix(embedding_type):
    y_df = data['Cat1_Cat2_Cat3']

    if embedding_type == 'TF_TF':
        tf_vectorizer = TfidfVectorizer(min_df=2)
        X_df = tf_vectorizer.fit_transform(data['cleaned_title_text'])

    if embedding_type == 'TF_W2V':
        sent_list = []
        for word_list in splitted_sent_list:
            sent_list.append(" ".join(word_list))
        data.loc[:, 'clean_title'] = sent_list[0:data.shape[0]]


        tf_vectorizer = TfidfVectorizer(min_df=2)
        X_title = tf_vectorizer.fit_transform(data['clean_title'])


        model = Word2Vec.load("word2vec_text.model")
        X_text = np.zeros(shape=(data.shape[0], 30))
        for index, row in data.iterrows():
            word_list = row['clean_text'].split()
            embedding = np.zeros(30)
            for word in word_list:
                embedding = np.add(embedding, model.wv[word])
            avg_embedding = embedding / len(word_list)
            X_text[index] = avg_embedding
        X_df = hstack([X_title, X_text])

    if embedding_type == 'TF_USE':
        tf_vectorizer = TfidfVectorizer(min_df=2)
        X_title = tf_vectorizer.fit_transform(data['clean_title'])
        X_text = pickle.load(open("reduced_USE_Text_embeddings_25.pkl", "rb"))
        X_df = hstack([X_title, X_text])

    if embedding_type == 'USE_USE':
        X_title = pickle.load(open("reduced_USE_Title_embeddings_25.pkl", "rb"))
        X_text = pickle.load(open("reduced_USE_Text_embeddings_25.pkl", "rb"))
        X_df = np.hstack([X_title, X_text])


    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=1,
                                                        stratify=y_df)

    return X_train, X_test, y_train, y_test



def all_models():
    all_embedding_types = ['TF_TF', 'TF_W2V', 'TF_USE', 'USE_USE']
    all_model_names = ['Logistic regression', 'KNN', 'SVM', 'Random_forest']
    for embedding_type in all_embedding_types:
        X_train, X_test, y_train, y_test = get_input_matrix(embedding_type)
        for model_name in all_model_names:
            parameters = dict()
            if model_name == 'Logistic regression':
                parameters = {'alpha': [10 ** x for x in range(-5, 1)]}
                model = SGDClassifier(penalty='l2', loss='log', class_weight='balanced',random_state=42)
            if model_name == 'KNN':
                parameters = {'n_neighbors': range(5, 25)}
                model = KNeighborsClassifier(weights='distance')
            if model_name == 'SVM':
                parameters = {'C': [10 ** x for x in range(1, 4)]}
                model = SVC(random_state=42, probability=True)
            if model_name == 'Random_forest':
                parameters = {'n_estimators': [100 * x for x in range(1, 5)]}
                model = RandomForestClassifier()

            clf = GridSearchCV(model, param_grid=parameters)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("Classification report for the model name:", model_name)
            print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    all_models()