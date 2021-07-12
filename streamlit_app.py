import streamlit as st

import PyPDF2 
from PyPDF2 import PdfFileWriter, PdfFileReader

import spacy
from spacy import displacy

import transformers
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer



#@st.cache
#def get_ner_pipeline() -> Pipeline:
#    ner = pipeline("ner", model="Luciano/bertimbau-large-lener_br", aggregation_strategy="average")
#    return ner
#
#pipeline = get_ner_pipeline()
#st.write(pipeline.model)
#st.write(pipeline.model.config)
#st.write(pipeline("Meu nome é Luciano Zanuz")




@st.cache
def carrega_modelo():
    nome_modelo_treinado = "Luciano/bertimbau-large-lener_br" # Modelo do Huggingface Hub
    modelo_treinado = AutoModelForTokenClassification.from_pretrained(nome_modelo_treinado)
    tokenizer_treinado = AutoTokenizer.from_pretrained(nome_modelo_treinado)
    return nome_modelo_treinado, modelo_treinado, tokenizer_treinado

colors = {"PESSOA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)",
          "TEMPO": "linear-gradient(0deg, rgba(34,193,195,1) 0%, rgba(253,187,45,1) 100%)",
          "LOCAL": "radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%)",
          "ORGANIZACAO": "linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%)",
          "LEGISLACAO": "radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%)",
          "JURISPRUDENCIA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)"
          }
options = {"colors": colors}

def mostra_ner(texto, aggregation_strategy):
    st.write('aqui')
    #nome_modelo, modelo, tokenizer = carrega_modelo()
    #st.write(nome_modelo, modelo, tokenizer)
    st.write('aqui2')
    ner = pipeline("ner", model=modelo, tokenizer=tokenizer, aggregation_strategy=aggregation_strategy)
    #ner = pipeline("ner", model=nome_modelo, aggregation_strategy=aggregation_strategy)
    #ner = pipeline("ner", aggregation_strategy=aggregation_strategy)
    #data = ner(texto)

    #ents = []
    #for item in data:
    #  item = {"label" if k == "entity_group" else k:v for k,v in item.items()}
    #  ents.append(item);

    #ex = [{"text": texto,
    #      "ents": ents,
    #      "title": None}]
    return texto
    #displacy.render(ex, style="ent", options=options, jupyter=True, manual=True)

st.write(st.session_state)

st.title('Reconhecimento de Entidades Nomeadas')
st.header('Header da aplicação.')
st.subheader('Subheader da aplicação')
st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

#container = st.beta_container()
#container.write("This is inside the container")
#st.write("This is outside the container")
#container.write("This is inside the container 2")
#
#def processa_pdf():
#    st.write('PDF processado')
#    st.sidebar.write('PDF processado sidebar')
#    container.write("This is inside the container 3")
#    return
#
#uploaded_file = st.sidebar.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
#if uploaded_file is not None:
#    bytes_data = uploaded_file.getvalue()
#    
#    # creating a pdf file object 
#    pdfFileObj = open(bytes_data, 'rb') 
#
#    # creating a pdf reader object 
#    pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
#
#    # printing number of pages in pdf file 
#    print(pdfReader.numPages)

#if st.sidebar.button('Enviar', key='bt_enviar'):
#   st.sidebar.write('Why hello there')
#else:
#   st.sidebar.write('Goodbye')
#
#st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)

#txt = st.text_area('Text to analyze', '''
#    It was the best of times, it was the worst of times, it was
#    the age of wisdom, it was the age of foolishness, it was
#    the epoch of belief, it was the epoch of incredulity, it
#    was the season of Light, it was the season of Darkness, it
#    was the spring of hope, it was the winter of despair, (...)
#    ''')

txt = st.text_area('Text to analyze', '''
A C Ó R D Ã O
Acordam os Senhores Desembargadores da 8ª TURMA CÍVEL do
Tribunal de Justiça do Distrito Federal e Territórios, Nídia Corrêa Lima -
Relatora, DIAULAS COSTA RIBEIRO - 1º Vogal, EUSTÁQUIO DE CASTRO - 2º
Vogal, sob a presidência do Senhor Desembargador DIAULAS COSTA RIBEIRO,
em proferir a seguinte decisão: RECURSO DE APELAÇÃO CONHECIDO E NÃO
PROVIDO. UNÂNIME., de acordo com a ata do julgamento e notas taquigráficas.
Brasilia(DF), 15 de Março de 2018.
''')

#mostra_ner(sequence, "simple")
#mostra_ner(sequence, "first")
#mostra_ner(sequence, "average")
#mostra_ner(sequence, "max")
st.subheader('Análise NER')
st.write(mostra_ner(txt, 'average'))
