import streamlit as st

import spacy
from spacy import displacy

import PyPDF2 
from PyPDF2 import PdfFileWriter, PdfFileReader

import transformers
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

import json
import requests

st.title('Reconhecimento de Entidades Nomeadas')
#st.header('Header da aplicação.')
#st.subheader('This model is a fine-tuned version of neuralmind/bert-large-portuguese-cased on the lener_br dataset')
#st.text('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')
#st.write('Carregue o arquivo de algum texto jurídico em PDF e clique em Enviar')

st.sidebar.subheader('This model is a fine-tuned version of neuralmind/bert-large-portuguese-cased on the lener_br dataset')
modelo = st.sidebar.radio(
    "Modelo treinado",
    ('Luciano/bertimbau-large-lener_br', 'Luciano/bertimbau-base-lener_br'))
API_URL = "https://api-inference.huggingface.co/models/" + modelo
API_TOKEN = st.secrets["api_token"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}

aggregation_strategy = st.sidebar.radio(
    "Aggregation strategy",
    ('simple', 'first', 'average', 'max'))

opt_txt_exemplo = st.sidebar.selectbox(
    'Texto de exemplo',
    ('Meu nome é Luciano', 'Meu nome é Juliano', 'Texto LeNER_BR', 'Texto TJRS', 'Op1', 'Op2'))
if(opt_txt_exemplo=="Meu nome é Luciano"):
    txt_exemplo = "Meu nome é Luciano Zanuz e eu moro em Porto Alegre, Rio Grande do Sul, Brasil."
elif(opt_txt_exemplo=="Meu nome é Juliano"):
    txt_exemplo = "Meu nome é Juliano Pacheco e eu moro em Canoas, Rio Grande do Sul, Brasil."
elif(opt_txt_exemplo=="Texto LeNER_BR"):
    txt_exemplo = '''A C Ó R D Ã O
Acordam os Senhores Desembargadores da 8ª TURMA CÍVEL do
Tribunal de Justiça do Distrito Federal e Territórios, Nídia Corrêa Lima -
Relatora, DIAULAS COSTA RIBEIRO - 1º Vogal, EUSTÁQUIO DE CASTRO - 2º
Vogal, sob a presidência do Senhor Desembargador DIAULAS COSTA RIBEIRO,
em proferir a seguinte decisão: RECURSO DE APELAÇÃO CONHECIDO E NÃO
PROVIDO. UNÂNIME., de acordo com a ata do julgamento e notas taquigráficas.
Brasilia(DF), 15 de Março de 2018.
'''
elif(opt_txt_exemplo=="Texto TJRS"):
    txt_exemplo = '''EGRÉGIO TRIBUNAL DE JUSTIÇA DO ESTADO DO RIO GRANDE DO SUL
REF.
AUTOS Nº : 5000307-41.2020.8.21.5001
OBJETO: AGRAVO DE INSTRUMENTO
FACTA FINANCEIRA S. A., inscrita no CNPJ sob o n°
15.581.638/0001-30, com sede na Rua dos Andradas nº 1409, 07º
andar – Bairro Centro, Porto Alegre/RS, CEP 90020-011, irresignada
com decisão proferida nos autos do processo nº: 5000307-
41.2020.8.21.5001, em trâmite no 1º Juízo da 2ª Vara Cível do Foro
Regional do Sarandi da Comarca de Porto Alegre/RS, intentado por
CLARINDA MARQUES SOARES, já qualificado nos autos, vem,
respeitosamente, com fulcro no artigo 1.015 do Novo Código de
Processo Civil, interpor tempestivamente o presente
                      AGRAVO DE INSTRUMENTO,
conforme as razões que seguem em anexo, requerendo, desde já, que
as mesmas sejam recebidas, processadas e levadas à apreciação de
uma de suas Colendas Câmaras.
Termos em que,
Pede deferimento.
Porto Alegre/RS, 17 de julho de 2020.
'''
else:
    txt_exemplo = ""    
    
debug = st.sidebar.checkbox('Debug')
if(debug):
    st.write(st.session_state)
    
### Processamento do pipeline

colors = {"PESSOA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)",
          "TEMPO": "linear-gradient(0deg, rgba(34,193,195,1) 0%, rgba(253,187,45,1) 100%)",
          "LOCAL": "radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(252,70,107,1) 100%)",
          "ORGANIZACAO": "linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%)",
          "LEGISLACAO": "radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%)",
          "JURISPRUDENCIA": "linear-gradient(90deg, rgba(9,2,124,1) 0%, rgba(34,34,163,1) 35%, rgba(0,212,255,1) 100%)"
          }
options = {"colors": colors}

def ner_pipeline(texto, modelo_treinado, tokenizer_treinado, aggregation_strategy):
    ner = pipeline("ner", model=modelo_treinado, tokenizer=tokenizer_treinado, aggregation_strategy=aggregation_strategy)
    data = ner(texto)

    ents = []
    for item in data:
      item = {"label" if k == "entity_group" else k:v for k,v in item.items()}
      ents.append(item);

    ex = [{"text": texto,
          "ents": ents,
          "title": None}]
    return displacy.render(ex, style="ent", options=options, manual=True)    

#nome_modelo_treinado = "Luciano/bertimbau-large-lener_br" 
nome_modelo_treinado = modelo
@st.cache
def carrega_modelo(nome_modelo_treinado):
    modelo_treinado = AutoModelForTokenClassification.from_pretrained(nome_modelo_treinado)
    return modelo_treinado
modelo_treinado = carrega_modelo(nome_modelo_treinado)

#@st.cache(hash_funcs={tokenizers.Tokenizer: my_hash_func})
@st.cache(allow_output_mutation=True) # Parâmetro necessário para não dar erro de hash
def carrega_tokenizer(nome_modelo_treinado):
    tokenizer_treinado = AutoTokenizer.from_pretrained(nome_modelo_treinado)
    return tokenizer_treinado
tokenizer_treinado = carrega_tokenizer(nome_modelo_treinado)

txt = st.text_area('Texto a ser analisado', txt_exemplo, height=300, key="area1")
st.write(ner_pipeline(txt, modelo_treinado, tokenizer_treinado, aggregation_strategy),unsafe_allow_html=True)

### API

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

txt = st.text_area('Texto a ser analisado', txt_exemplo, height=300, key="area2")
if(debug):
    data = query({"inputs": txt})
    if("error" in data):
        st.write(data["error"])
    st.write(data)
    data = ajusta_retorno_api(data)
    st.write(data)
st.write(mostra_ner(txt, ajusta_retorno=True),unsafe_allow_html=True)

### Leitura de PDF

st.write("aqui-1")
uploaded_file = st.file_uploader("Selecione um PDF", help="Selecione um arquivo em PDF referente a uma petição ou texto jurídico.")
pdf_text = ""
if uploaded_file is not None:
    pdfReader = PyPDF2.PdfFileReader(uploaded_file) 
    for page in pdfReader.pages:
        pdf_text += page.extractText()
        #st.write(page.extractText())
st.write(pdf_text)
txt = st.text_area('Texto a ser analisado', pdf_text, height=300)
st.write("aqui-2")

if(debug):
    data = query({"inputs": txt})
    if("error" in data):
        st.write(data["error"])
    st.write(data)
    data = ajusta_retorno_api(data)
    st.write(data)
st.write(mostra_ner(txt, ajusta_retorno=True),unsafe_allow_html=True)

        



#if st.sidebar.button('Enviar', key='bt_enviar'):
#   st.sidebar.write('Why hello there')
#else:
#   st.sidebar.write('Goodbye')
#
#st.sidebar.button('Enviar click', key='bt_enviar_click', on_click=processa_pdf)
