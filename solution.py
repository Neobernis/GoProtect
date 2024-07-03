import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sentence_transformers import SentenceTransformer
from utils import (cleaning_name,
                   find_scool_name,
                   load_model,
                   load_embeddings,
                   convert_df)



model = load_model()

st.title('Определения эталонного наименования школы по варианту')
school_name = st.text_input('Введите регион и название школы:', placeholder='Например, Москва Триумф')

with st.sidebar.header('Загрузите CVS файл  '):
    uploaded_file = st.sidebar.file_uploader('Названия школ должны находиться в поле name', type=["csv"])

@st.cache_data
def load_reference():
    reference_schools = pd.read_csv('data_02_match_school/reference_schools.csv')
    return reference_schools
reference_schools = load_reference()
corpus_embeddings = load_embeddings()

if len(school_name) > 0:
    clean_name = cleaning_name(school_name)
    ans_input = find_scool_name(clean_name, model, reference_schools, corpus_embeddings) 
    st.subheader('Результат')
    st.table(ans_input)
    
if uploaded_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    clean_df_name = cleaning_name(df['name'])
    ans_df = find_scool_name(clean_df_name, model, reference_schools, corpus_embeddings) 
    st.subheader('Результат CSV')
    
    csv_download = convert_df(ans_df)
    st.download_button(
        label="Download data as CSV",
        data=csv_download,
        file_name="comparison_scools.csv",
        mime="text/csv")
    
    st.table(ans_df)