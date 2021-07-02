import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

'''Loading the data and embeddings'''
data = pd.read_csv("data.csv")
text_sentence_embeddings_dict = pickle.load(open("text_sentence_embedding.pkl", "rb"))
title_sentence_embeddings_dict = pickle.load(open("title_sentence_embedding.pkl", "rb"))




print("shape of the dataset:", data.shape)


'''Handling missing values and duplicates and preprocessing text'''
data.fillna('NA', inplace=True)
print(data.info())
data.drop_duplicates(subset=['Title', 'Text'], inplace=True, ignore_index=True)
data.Text = data.Text.str.replace(r'[^a-zA-Z0-9\n]', ' ')
data.Text = data.Text.str.replace(r'\s+', ' ')
data.Title = data.Title.str.replace(r'[^a-zA-Z0-9\n]', ' ')
data.Title = data.Title.str.replace(r'\s+', ' ')

def getting_embedding_matrix():
    text_sentence_embeddings = []
    title_sentence_embeddings = []
    for index, row in data.iterrows():
      text_sentence_embeddings.append(text_sentence_embeddings_dict[row['Text']])
      title_sentence_embeddings.append(title_sentence_embeddings_dict[row['Title']])

    text_sentence_embeddings = np.array(text_sentence_embeddings)
    title_sentence_embeddings = np.array(title_sentence_embeddings)
    print(text_sentence_embeddings.shape)
    print(type(text_sentence_embeddings))
    print(title_sentence_embeddings.shape)
    print(type(title_sentence_embeddings))
    X_title = title_sentence_embeddings
    X_text = text_sentence_embeddings
    X_df = np.hstack([X_title, X_text])
    return X_df



def calculating_performance_matrix(X_test):
    model1 = pickle.load(open("level_1_rf_UU_model.pkl", 'rb'))
    model2 = pickle.load(open("level_2_rf_UU_model.pkl", 'rb'))
    model3 = pickle.load(open("level_3_rf_UU_model.pkl", 'rb'))
    y_df = data.iloc[X_test.index, :].loc[:, ['Cat1', 'Cat2', 'Cat3']]
    y_true = []
    for index, row in y_df.iterrows():
        y_true.append([row['Cat1'], row['Cat2'], row['Cat3']])


    y_pred1 = model1.predict(X_test)

    X_test_2 = np.stack([X_test, y_pred1])
    y_pred2 = model2.predict(X_test_2)

    X_test_3 = np.stack([X_test_2, y_pred2])
    y_pred3 = model3.predict(X_test_3)
    y_pred = []
    sum = 0
    for i in range(len(y_true)):
        sum = sum + len(list(set(y_true[i]) & set(y_pred[i])))/3

    print("Accuracy of the predictions:", sum/len(y_true))

    '''Precision recall and F1 score implementation'''















# def dimensionality_reduction():
#     '''Reducing the dimension of the embeddings using UMAP'''
#
#     # X_title = pickle.load(open("reduced_USE_Title_embeddings_25.pkl", "rb"))
#     # X_text = pickle.load(open("reduced_USE_Text_embeddings_25.pkl", "rb"))
#     # print(X_text.shape)
#     # print(type(X_text))
#     # print(X_title.shape)
#     # print(type(X_title))
#     X_df = np.hstack([X_title, X_text])
#     return X_df




def level_1_classifier(X_df, train_indices, test_indices):
    "Level 1 classifier"

    y_df = data['Cat1']
    X_df = pd.DataFrame(X_df)
    X_train = X_df.iloc[train_indices, :]
    X_test = X_df.iloc[test_indices, :]
    y_train = y_df.iloc[train_indices]
    y_test = y_df.iloc[test_indices]

    # parameters = {'n_estimators': [100 * x for x in range(1, 5)]}
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced')
    # clf = GridSearchCV(random, param_grid=parameters, scoring='neg_log_loss', return_train_score=True)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    # print(clf.best_params_)
    pickle.dump(clf, open("level_1_rf_UU_model.pkl", 'wb'))
    print('Done')

def level_2_classifier(X_df, train_indices, test_indices):
    '''Level 2 classifier'''

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_level1 = pd.DataFrame(enc.fit_transform(data[['Cat1']]).toarray())
    X_df = np.hstack([X_df, enc_level1])
    X_df = pd.DataFrame(X_df)
    print(X_df.shape)
    print(type(X_df))
    print(enc_level1.shape)
    print(type(enc_level1))
    y_df = data['Cat2']
    X_train = X_df.iloc[train_indices, :]
    X_test = X_df.iloc[test_indices, :]
    y_train = y_df.iloc[train_indices]
    y_test = y_df.iloc[test_indices]
    # parameters = {'n_estimators': [100 * x for x in range(1, 5)]}
    clf = RandomForestClassifier(n_estimators=400, class_weight='balanced')
    # clf = GridSearchCV(random, param_grid=parameters)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    # print(clf.best_params_)
    pickle.dump(clf, open("level_2_rf_UU_model.pkl", 'wb'))
    return X_df


def level_3_classifier(X_df, train_indices, test_indices):
    '''Level 3 Classifier'''
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_level2 = pd.DataFrame(enc.fit_transform(data[['Cat2']]).toarray())
    print(X_df.shape)
    print(type(X_df))
    print(enc_level2.shape)
    print(type(enc_level2))
    X_df = np.hstack([X_df, enc_level2])
    X_df = pd.DataFrame(X_df)
    y_df = data['Cat3']
    X_train = X_df.iloc[train_indices, :]
    X_test = X_df.iloc[test_indices, :]
    y_train = y_df.iloc[train_indices]
    y_test = y_df.iloc[test_indices]
    # parameters = {'n_estimators': [100 * x for x in range(1, 5)]}
    clf = RandomForestClassifier(n_estimators=400, class_weight='balanced')
    # clf = GridSearchCV(random, param_grid=parameters)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    # print(clf.best_params_)
    pickle.dump(clf, open("level_3_rf_UU_model.pkl", 'wb'))
    print(classification_report(y_test, y_pred_test))

if __name__ == "__main__":
    X_df = getting_embedding_matrix()
    all_indices = list(range(len(data)))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2)
    # pickle.dump(train_indices, open("train_indices.pkl", 'wb'))
    # pickle.dump(test_indices, open("test_indices.pkl", 'wb'))
    level_1_classifier(X_df, train_indices, test_indices)
    X_df = level_2_classifier(X_df, train_indices, test_indices)
    level_3_classifier(X_df, train_indices, test_indices)