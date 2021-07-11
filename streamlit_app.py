import pandas as pd
import numpy as np
import streamlit as st

st.write(st.session_state)

st.title('Reconhecimento de Entidades Nomeadas')
st.header('Header da aplicação.')
st.subheader('Subheader da aplicação')
st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

def processa_pdf():
    st.write('PDF processado')
    st.sidebar.write('PDF processado sidebar')
    container.write("PDF processado no container")
    return

file_match = st.sidebar.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
if st.sidebar.button('Enviar', key='bt_enviar'):
   st.sidebar.write('Why hello there')
else:
   st.sidebar.write('Goodbye')

st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)

container = st.beta_container()
container.write("This is inside the container")

st.write("This is outside the container")

@st.cache
def fetch_and_clean_data():
    df = pd.read_csv('<some csv>')
    # do some cleaning
    return df
