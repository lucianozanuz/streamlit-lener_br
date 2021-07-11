import pandas as pd
import numpy as np
import streamlit as st

st.write(st.session_state)

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
    st.sidebar.write('PDF processado sidebar')
    return

file_match = st.sidebar.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
if st.sidebar.button('Enviar', key='bt_enviar'):
   st.sidebar.write('Why hello there')
   bt_enviar=false
else:
   st.sidebar.write('Goodbye')

st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)
   
@st.cache
def fetch_and_clean_data():
    df = pd.read_csv('<some csv>')
    # do some cleaning
    return df
