import streamlit as st

import spacy
from spacy import displacy

import PyPDF2 
from PyPDF2 import PdfFileWriter, PdfFileReader

import transformers
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

import json
import requests

debug = st.checkbox('Debug')
if(debug):
    st.write(st.session_state)

st.title('Reconhecimento de Entidades Nomeadas')
#st.header('Header da aplicação.')
st.subheader('This model is a fine-tuned version of neuralmind/bert-large-portuguese-cased on the lener_br dataset')
st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')
#st.write('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

modelo = st.radio(
    "Modelo treinado",
    ('Luciano/bertimbau-large-lener_br', 'Luciano/bertimbau-base-lener_br'))
API_URL = "https://api-inference.huggingface.co/models/" + modelo
API_TOKEN = st.secrets["api_token"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def ajusta_retorno_api(data):
  new_data = []
  new_i = -1
  for i, item in enumerate(data):
    if(item["word"][:2] == "##"):
      new_data[new_i]["word"] += item["word"][2:]
      new_data[new_i]["end"] = item["end"]
    else:
      new_data.append(item)
      new_i +=1
  return new_data

colors = {"PESSOA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)",
          "TEMPO": "linear-gradient(0deg, rgba(34,193,195,1) 0%, rgba(253,187,45,1) 100%)",
          "LOCAL": "radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%)",
          "ORGANIZACAO": "linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%)",
          "LEGISLACAO": "radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%)",
          "JURISPRUDENCIA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)"
          }
options = {"colors": colors}

def mostra_ner(texto, ajusta_retorno=False):
    #data = query({"inputs": texto, "options": {"wait_for_model": "true"}})
    data = query({"inputs": texto})
    if("error" in data):
        return data["error"]
    #"error":"Model Luciano/bertimbau-large-lener_br is currently loading"
                    
    if(ajusta_retorno):
      data = ajusta_retorno_api(data)
    
    ents = []
    for item in data:
      item = {"label" if k == "entity_group" else k:v for k,v in item.items()}
      ents.append(item);

    ex = [{"text": texto,
          "ents": ents,
          "title": None}]
    return displacy.render(ex, style="ent", options=options, manual=True)    

#txt = "Meu nome é Luciano Zanuz e eu moro em Porto Alegre, Rio Grande do Sul, Brasil."
#data = query(txt)
#st.write(data)
#txt = "Meu nome é Juliano Pacheco e eu moro em Canoas, Rio Grande do Sul, Brasil."
#data = query(txt)
#st.write(data)
#data = ajusta_retorno_api(data)
#st.write(data)
#st.write(mostra_ner(txt, ajusta_retorno=False),unsafe_allow_html=True)
#st.write(mostra_ner(txt, ajusta_retorno=True),unsafe_allow_html=True)

txt = st.text_area('Texto a ser analisado', '''A C Ó R D Ã O
Acordam os Senhores Desembargadores da 8ª TURMA CÍVEL do
Tribunal de Justiça do Distrito Federal e Territórios, Nídia Corrêa Lima -
Relatora, DIAULAS COSTA RIBEIRO - 1º Vogal, EUSTÁQUIO DE CASTRO - 2º
Vogal, sob a presidência do Senhor Desembargador DIAULAS COSTA RIBEIRO,
em proferir a seguinte decisão: RECURSO DE APELAÇÃO CONHECIDO E NÃO
PROVIDO. UNÂNIME., de acordo com a ata do julgamento e notas taquigráficas.
Brasilia(DF), 15 de Março de 2018.''', height=300)
if(debug):
    data = query({"inputs": txt})
    if("error" in data):
        st.write(data["error"])
    st.write(data)
    data = ajusta_retorno_api(data)
    st.write(data)
st.write(mostra_ner(txt, ajusta_retorno=True),unsafe_allow_html=True)





container = st.beta_container()
#container.write("This is inside the container")
#st.write("This is outside the container")
#container.write("This is inside the container 2")

def processa_pdf():
    #st.write('PDF processado')
    #container.write("This is inside the container 3")
    return
processa_pdf()

uploaded_file = st.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
pdf_text = ""
if uploaded_file is not None:
    pdfReader = PyPDF2.PdfFileReader(uploaded_file) 
    for page in pdfReader.pages:
        pdf_text += page.extractText()
        #st.write(page.extractText())
st.write(pdf_text)
txt = st.text_area('Texto a ser analisado', pdf_text, height=300)

if(debug):
    data = query({"inputs": txt})
    if("error" in data):
        st.write(data["error"])
    st.write(data)
    data = ajusta_retorno_api(data)
    st.write(data)
st.write(mostra_ner(txt, ajusta_retorno=True),unsafe_allow_html=True)







#@st.cache
#def get_ner_pipeline():
#    ner = pipeline("ner", model="Luciano/bertimbau-large-lener_br", aggregation_strategy="average")
#    return ner
#
#pipeline = get_ner_pipeline()
#st.write(pipeline.model)
#st.write(pipeline.model.config)
#st.write(pipeline("Meu nome é Luciano Zanuz"))





st.write("aqui-1")
@st.cache
def carrega_modelo():
    nome_modelo_treinado = "Luciano/bertimbau-large-lener_br" # Modelo do Huggingface Hub
    modelo_treinado = ""
    tokenizer_treinado = ""
    #modelo_treinado = AutoModelForTokenClassification.from_pretrained(nome_modelo_treinado)
    tokenizer_treinado = AutoTokenizer.from_pretrained(nome_modelo_treinado)
    return nome_modelo_treinado, modelo_treinado, tokenizer_treinado
st.write("aqui-2")
nome_modelo_treinado, modelo_treinado, tokenizer_treinado = carrega_modelo()
st.write("aqui-3")
st.write(nome_modelo_treinado)
st.write(modelo_treinado)
st.write(tokenizer_treinado)


#mostra_ner(sequence, "simple")
#mostra_ner(sequence, "first")
#mostra_ner(sequence, "average")
#mostra_ner(sequence, "max")
#st.subheader('Análise NER')
#st.write(mostra_ner(txt, 'average'))

        
#if st.sidebar.button('Enviar', key='bt_enviar'):
#   st.sidebar.write('Why hello there')
#else:
#   st.sidebar.write('Goodbye')
#
#st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)
