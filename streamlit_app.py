import pandas as pd
import numpy as np
import streamlit as st

st.write(st.session_state)

st.title('Reconhecimento de Entidades Nomeadas')
st.header('Header da aplicação.')
st.subheader('Subheader da aplicação')
st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

container = st.beta_container()
container.write("This is inside the container")
st.write("This is outside the container")
container.write("This is inside the container 2")

def processa_pdf():
    st.write('PDF processado')
    st.sidebar.write('PDF processado sidebar')
    container.write("This is inside the container 3")
    return

uploaded_file = st.sidebar.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
if uploaded_file is not None:
   # To read file as bytes:
   bytes_data = uploaded_file.getvalue()
   st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

#if st.sidebar.button('Enviar', key='bt_enviar'):
#   st.sidebar.write('Why hello there')
#else:
#   st.sidebar.write('Goodbye')

st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)

@st.cache
def fetch_and_clean_data():
    df = pd.read_csv('<some csv>')
    # do some cleaning
    return df
