import pandas as pd
import pickle
data = pd.read_csv("data.csv")
print("shape of the dataset:", data.shape)
data.head()
data.fillna('NA', inplace=True)


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sentences = data.Title.to_list()
sentence_embeddings = model.encode(sentences)
sentence_embedding_dict = dict()
for sentence, embedding in zip(sentences, sentence_embeddings):
    sentence_embedding_dict[sentence] = embedding
title_sentence_embeddings = pickle.load(open("USE_USE/title_sentence_embedding.pkl", "rb"))