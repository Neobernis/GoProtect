import streamlit as st
import pandas as pd
import numpy as np
from torch import topk
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

@st.cache_data
def load_embeddings():
    corpus_embeddings = np.load('model/corpus_embeddings.npy')
    return corpus_embeddings

@st.cache_data
def cleaning_name(name):
    words_to_replace = ['No', 'ООО', 'ГБУСО', 'ФСО', 'ДО', 'МБУ', 'ОР', 'МАУ', 'ДЮЦ', 'АНО', 'ГБУ', 'ГУ', 'ТО', 'ГАУ', 'БУ', 'ГБПОУ']
    if type(name) == str:
        clean_name = name.split(',')
    clean_name = pd.Series(name, name='Имя школы')
    clean_name =  (clean_name.replace(r'[^А-Яа-яёЁA-Za-z\s\d+]', ' ', regex=True)
                       .replace(words_to_replace, '', regex=True)
                       .replace('ё', 'е')
                       .replace(r'\s+', ' ', regex=True)
                       .str.strip())
    return clean_name

def find_scool_name(clean_name, model, reference_schools, corpus_embeddings):
    ans = []
    for i, query in enumerate(list(clean_name)):
        query_embedding = model.encode(query, convert_to_tensor=True)
        similarity_scores = model.similarity(query_embedding, corpus_embeddings)[0]
        scores, indices = topk(similarity_scores, k=1)

        for score, idx in zip(scores, indices):
            ans.append({'query': query, 
                        'corpus_name': str(reference_schools.iloc[int(idx)][3]) + ' ' + str(reference_schools.loc[int(idx)][2]), 
                        'school_id': reference_schools.iloc[int(idx)][0], 
                        'score': float(score),})
    return ans

@st.cache_data
def convert_df(df):
    df_frame = pd.DataFrame(df)
    return df_frame.to_csv().encode("utf-8")