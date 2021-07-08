import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model1 = pickle.load(open("level_1_rf_UU_model.pkl", 'rb'))
model2 = pickle.load(open("level_2_rf_UU_model.pkl", 'rb'))
model3 = pickle.load(open("level_3_rf_UU_model.pkl", 'rb'))
print('Models loaded')

def generate_embeddings(title, text):
    print('entering in ')
    title = title.replace(r'[^a-zA-Z0-9\n]', ' ')
    title = title.replace(r'\s+', ' ')
    text = text.replace(r'[^a-zA-Z0-9\n]', ' ')
    text = text.replace(r'\s+', ' ')
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    title_embeddings = model.encode(title)
    text_embeddings = model.encode(text)
    X_test = np.hstack([title_embeddings, text_embeddings])
    return X_test



def predict(title, text):

    X_test = generate_embeddings(title, text)
    y_pred1 = model1.predict([X_test])
    y_prob1 = model1.predict_proba([X_test])[0]
    index = np.argmax(y_prob1)
    onehot_encoding1 = np.zeros(len(y_prob1))
    onehot_encoding1[index] = 1

    X_test_2 = np.hstack([X_test, onehot_encoding1])
    y_pred2 = model2.predict([X_test_2])
    y_prob2 = model2.predict_proba([X_test_2])[0]
    index = np.argmax(y_prob2)
    onehot_encoding2 = np.zeros(len(y_prob2))
    onehot_encoding2[index] = 1

    X_test_3 = np.hstack([X_test_2, onehot_encoding2])
    y_pred3 = model3.predict([X_test_3])
    y_prob3 = model3.predict_proba([X_test_3])[0]
    index = np.argmax(y_prob3)
    onehot_encoding3 = np.zeros(len(y_prob3))
    onehot_encoding3[index] = 1

    result = [y_pred1[0], y_pred2[0], y_pred3[0]]
    return {"Cat1": result[0], "Cat2": result[1], "Cat3": result[2]}





if __name__ == "__main__":
    title = "Joy Honey & Almonds Advanced Nourishing Body Lotion, For Normal to Dry skin"
    text = "The lotion inside the bottle is not the same quality as the make of Joy. It clearly seemed much more less creamy and not smelling good at all."

    result = predict(title, text)
    print(result)










