import nltk
import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import pickle
import numpy as np
from scipy.sparse import hstack

model1 = pickle.load(open("Models/level_1_LinSVC_TT_model.pkl", 'rb'))
enc1 = pickle.load(open("onehotencoder_level1.pkl", "rb"))
enc2 = pickle.load(open("onehotencoder_level2.pkl", "rb"))
tf_vectorizer = pickle.load(open("tf_idf_model.pkl", "rb"))
model3 = pickle.load(open("Models/level_3_LinSVC_model.pkl", "rb"))

def text_preprocessing(Title, Text):
    '''This function cleans text, remove stopwords and symbols, lowercase the text
         and makes tf_idf embedding matrix using the title and text columns in the data file'''

    stop_words = set(stopwords.words('english'))

    total_text = Title + " " + Text
    sentence = " "
    if type(total_text) is str:
        if type(total_text) is not int:
            total_text = re.sub(r'[^a-zA-Z0-9\n]', " ", total_text).lower()
            total_text = re.sub(r'\s+', " ", total_text)
            word_list = total_text.split()

            word_list = [word for word in word_list if word not in stop_words]
            word_list = [word for word in word_list if len(word) > 1]
            sentence = " ".join(word_list)
    else:
        print("there is no text description for id:")



    X_test = tf_vectorizer.transform([sentence])
    return X_test


def predict(title, text):
    '''Input:
    title: tilte of the product
    text: comment text
    output: returns a dictionary of predictions for category1, category2, category3
    Function: This function predicts the output for a given title and text'''
    X_test = text_preprocessing(title, text)

    y_pred1 = model1.predict(X_test)
    onehot_encoding1 = enc1.transform([[y_pred1[0]]])

    model2 = pickle.load(open("Models/" + str(y_pred1[0]) + "_level_2_LinSVC_model.pkl", "rb"))
    y_pred2 = model2.predict(X_test)
    onehot_encoding2 = enc2.transform([[y_pred2[0]]])

    X_test = hstack([X_test, onehot_encoding1])
    X_test = hstack([X_test, onehot_encoding2])

    y_pred3 = model3.predict(X_test)

    result = [y_pred1[0], y_pred2[0].split("$")[1], y_pred3[0]]
    return {"Cat1": result[0], "Cat2": result[1], "Cat3": result[2]}





if __name__ == "__main__":
    title = "Joy Honey & Almonds Advanced Nourishing Body Lotion, For Normal to Dry skin"
    text = "The lotion inside the bottle is not the same quality as the make of Joy. It clearly seemed much more less creamy and not smelling good at all."
    data = pd.DataFrame([title])
    result = predict(title, text)
    print(result)










