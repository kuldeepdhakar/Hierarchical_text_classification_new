import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
print("shape of the dataset:", data.shape)
data.fillna('NA', inplace=True)
data.drop_duplicates(subset=['Title', 'Text'], inplace=True, ignore_index=True)

'''Following code snippet replaces the less frequent classes with a common label in category 2 and 3'''
value_counts_l2 = data['Cat2'].value_counts()
value_counts_l3 = data['Cat3'].value_counts()
for index, row in data.iterrows():
    if value_counts_l3[row['Cat3']] < 5:
        data['Cat3'].iloc[index] = 'Lower_count_l3'
#############################################################################################

data['Cat1_Cat2'] = data['Cat1'] + "$" + data['Cat2']

# def tf_idf_w2v(df):
#     list_of_sentences = [sent.split() for sent in df['clean_title_text'].tolist()]
#     # TF-IDF weighted
#     w2v_model = Word2Vec.load("word2vec.model")
#     tf_idf_model = pickle.load(open("tf_idf_model.pkl", "rb"))
#     tfidf_feat = tf_idf_model.get_feature_names()  # tfidf words/col-names
#     dictionary = dict(zip(tf_idf_model.get_feature_names(), list(tf_idf_model.idf_)))
#     # final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf
#     tfidf_sent_vectors = []  # the tfidf-w2v for each sentence/review is stored in this list
#     row = 0
#     w2v_words = list(w2v_model.wv.key_to_index.keys())
#     for sent in tqdm(list_of_sentences):  # for each review/sentence
#         sent_vec = np.zeros(64)  # as word vectors are of zero length
#         weight_sum = 0  # num of words with a valid vector in the sentence/review
#         for word in sent:  # for each word in a review/sentence
#             if word in w2v_words and word in tfidf_feat:
#                 vec = w2v_model.wv[word]
#                 # tf_idf = tf_idf_matrix[row, tfidf_feat.index(word)]
#                 # to reduce the computation we are
#                 # dictionary[word] = idf value of word in whole courpus
#                 # sent.count(word) = tf valeus of word in this review
#                 tf_idf = dictionary[word] * (sent.count(word) / len(sent))
#                 sent_vec += (vec * tf_idf)
#                 weight_sum += tf_idf
#         if weight_sum != 0:
#             sent_vec /= weight_sum
#         tfidf_sent_vectors.append(sent_vec)
#         row += 1
#     return np.array(tfidf_sent_vectors)


# def word2vec_embeddings(train):
#     sent_list = [sent.split() for sent in train['clean_title_text'].tolist()]
#     model = Word2Vec(sentences=sent_list, vector_size=64, window=5, min_count=1, workers=4)
#     model.save("word2vec.model")




def text_preprocessing():
    '''This function cleans text, remove stopwords and symbols, lowercase the text
     and makes tf_idf embedding matrix using the title and text columns in the data file'''
    stop_words = set(stopwords.words('english'))
    splitted_sent_list = []

    data['title_text'] = data['Title'] + " " + data['Text']
    for index, row in data.iterrows():
        if type(row['title_text']) is str:
            total_text = row['title_text']
            if type(total_text) is not int:
                total_text = re.sub(r'[^a-zA-Z0-9\n]', " ", total_text).lower()
                total_text = re.sub(r'\s+', " ", total_text)
                word_list = total_text.split()


                word_list = [word for word in word_list if word not in stop_words]
                word_list = [word for word in word_list if len(word) > 1]
                sentence = " ".join(word_list)
                splitted_sent_list.append(sentence)
        else:
            print("there is no text description for id:", index)

    data['clean_title_text'] = splitted_sent_list
    train, test = train_test_split(data, test_size=0.2, random_state=1, stratify=data['Cat2'])
    tf_vectorizer = TfidfVectorizer(min_df=2)
    X_train = tf_vectorizer.fit_transform(train['clean_title_text'])
    X_test = tf_vectorizer.transform(test['clean_title_text'])

    pickle.dump(tf_vectorizer, open("tf_idf_model.pkl", "wb"))
    return X_train, X_test, train, test


def training_level1_classifier(X_train, X_test, train, test):
    '''
    Inputs:
    X_train: TF-IDF embedding matrix for training set
    X_test: TF-IDF embedding matrix for testing set
    train: Training dataframe
    test: test dataframe

    Function: This function trains a Flat classifier for category one using the TF-IDF embeddings and
    LinearSVC as an algorithm
    '''

    y_train = train['Cat1']
    y_test = test['Cat1']
    parameters = {'C': [10 ** x for x in range(0, 5)], 'penalty': ['l2', 'l1']}
    linsvc = LinearSVC(dual=False)
    clf = GridSearchCV(linsvc, param_grid=parameters)
    clf = CalibratedClassifierCV(clf)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    pickle.dump(clf, open("Models/level_1_LinSVC_TT_model.pkl", 'wb'))


def training_level2_classifier(X_train, X_test, train, test):
    '''
        Inputs:
        X_train: TF-IDF embedding matrix for training set
        X_test: TF-IDF embedding matrix for testing set
        train: Training dataframe
        test: test dataframe

        Function: This function trains Node based classifiers for category two using the TF-IDF embeddings and
        LinearSVC as an algorithm
        '''
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    cat1_values = data['Cat1'].unique().tolist()
    for category in cat1_values:
        df_train = train[train['Cat1'] == category]
        X_train_new = X_train[df_train.index, :]
        df_test = test[test['Cat1'] == category]
        X_test_new = X_test[df_test.index, :]
        y_train = df_train['Cat1_Cat2']
        y_test = df_test['Cat1_Cat2']


        parameters = {'C': [10 ** x for x in range(0, 5)], 'penalty': ['l2', 'l1']}
        linsvc = LinearSVC(dual=False)
        clf = GridSearchCV(linsvc, param_grid=parameters)
        clf = CalibratedClassifierCV(clf)
        clf.fit(X_train_new, y_train)
        y_pred = clf.predict(X_test_new)
        print("Classification report for category:", category)
        print(classification_report(y_test, y_pred))
        # saving the model
        pickle.dump(clf, open("Models/" + str(category) + "_level_2_LinSVC_model.pkl", 'wb'))


def training_level3_classifier(X_train, X_test, train, test):
    '''
        Inputs:
        X_train: TF-IDF embedding matrix for training set
        X_test: TF-IDF embedding matrix for testing set
        train: Training dataframe
        test: test dataframe

        Function: This function trains a Flat classifier for category three using the TF-IDF embeddings
        and the onehot encodings of the first and second categories and
        LinearSVC as an algorithm
        '''
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_level1 = pd.DataFrame(enc.fit_transform(train[['Cat1']]).toarray())
    pickle.dump(enc, open("onehotencoder_level1.pkl", "wb"))
    X_train = hstack([X_train, enc_level1])
    enc_level1 = pd.DataFrame(enc.transform(test[['Cat1']]).toarray())
    X_test = hstack([X_test, enc_level1])
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_level2 = pd.DataFrame(enc.fit_transform(train[['Cat1_Cat2']]).toarray())
    pickle.dump(enc, open("onehotencoder_level2.pkl", "wb"))
    X_train = hstack([X_train, enc_level2])
    enc_level2 = pd.DataFrame(enc.transform(test[['Cat1_Cat2']]).toarray())
    X_test = hstack([X_test, enc_level2])


    y_train = train['Cat3']
    y_test = test['Cat3']
    parameters = {'C': [10 ** x for x in range(0, 5)], 'penalty': ['l2', 'l1']}
    linsvc = LinearSVC(class_weight='balanced')
    clf = GridSearchCV(linsvc, param_grid=parameters)
    clf = CalibratedClassifierCV(clf)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    # saving the model
    pickle.dump(clf, open("Models/level_3_LinSVC_model.pkl", 'wb'))


if __name__ == "__main__":
    X_train, X_test, train, test = text_preprocessing()
    # word2vec_embeddings(train)
    # X_train = tf_idf_w2v(train)
    # X_test = tf_idf_w2v(test)
    print(X_train.shape)
    print(X_test.shape)
    training_level1_classifier(X_train, X_test, train, test)
    training_level2_classifier(X_train, X_test, train, test)
    training_level3_classifier(X_train, X_test, train, test)
    print('done')