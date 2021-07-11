import pandas as pd
import numpy as np
import streamlit as st

st.title('Reconhecimento de Entidades Nomeadas')
st.header('Header da aplicação.')
st.subheader('Subheader da aplicação')
st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

#st.sidebar.title('Reconhecimento de Entidades Nomeadas')
#st.sidebar.header('Header da aplicação.')
#st.sidebar.subheader('Subheader da aplicação')
#st.sidebar.text('Texto: Upload excel files with only one column, even if you put multiple columns only the first one will be used')

def processa_pdf():
    st.write('PDF processado')
    st.write('PDF processado sidebar')

file_match = st.sidebar.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
if st.sidebar.button('Enviar'):
   st.sidebar.write('Why hello there')
   st.sidebar.write('Why hello there')
else:
   st.sidebar.write('Goodbye')

if st.sidebar.button('Enviar cli', on_click='processa_pdf'):
   st.sidebar.write('Why hello there')
else:
   st.sidebar.write('Goodbye')

   
@st.cache
def fetch_and_clean_data():
    df = pd.read_csv('<some csv>')
    # do some cleaning
    return df


# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
