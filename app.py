"""
Código para criação de chatbot usando banco de dados vetorial e LLM para interação com documento PDF
"""
#carregamento de chaves de API e TOKEN do SECRETS do Streamlit

import streamlit as st
LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]

#PASSO 1 - CARREGAMENTO DE PDF
#montando o drive do Google Drive de onde lerá o dataset.
arquivo = "Ato_Requisitos_Tecnicos_Satelites.pdf"

#PASSO 2 - PARTICIONAMENTO DE DOCUMENTOS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#extrai texto do PDF e carrega o conteúdo p/ uso posterior em pipeline IA com RAG
from langchain_community.document_loaders import PyPDFLoader
dados = PyPDFLoader(arquivo).load()

#divide texto em "chunks" menores e armazena na variável "textos"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300) #400 e 100 originalmente
textos = text_splitter.split_documents(dados)

# PASSO 3 - EMBEDDINGS
#Embedding usando um modelo disponível no Hugging Face de Text Vector Embedding.
#Python framework for state-of-the-art sentence, text and image embeddings.
#SentenceTransformer é a biblioteca e precisamos escolher um modelo para realizar o processo de geração dos feature vectors.
#abaixo é criada a engine associada para calcular os feature vectors.
from langchain_huggingface import HuggingFaceEmbeddings
embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# PASSO 4 - VECTOR DATABASE com FAISS (para uso no Streamlit)
from langchain_community.vectorstores import FAISS
vector_db = FAISS.from_documents(textos, embedding_engine)

# Para buscar documentos similares
retriever = vector_db.as_retriever()

# Specify a persistence directory (passo criado pois estava dando um erro no db vetorial, dica GEMINI)
#persist_directory = "db"  # Choose a suitable directory name
# vector_db = Chroma.from_documents(textos, embedding_engine) comentado devido a erros
# cria bd vetorial vector database chamado ChromaDB.
#vector_db = Chroma.from_documents(textos, embedding_engine, persist_directory=persist_directory)

# Definir variaveis para chaves HF_TOKEN, OPENAI_API_KEY e LANGSMITH_API_KEY;

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
#LANGSMITH_API_KEY = userdata.get('LANGSMITH_API_KEY')
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

n_documentos = 15

def format_docs(documentos):
    return "\n\n".join(documento.page_content for documento in documentos)

#PASSO 8 
#usar os docs obtidos no vector database como contexto para nosso pedido de informação ao LLM.
#precisamos de um prompt "especial" que contextualiza o pedido por informação com os documentos.

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model = "gpt-4o-mini", temperature = 0)

rag = (
    {
        "question": RunnablePassthrough(),
        "context": vector_db.as_retriever(k = n_documentos)
                    | format_docs
    }
    | prompt
    | llm
    | StrOutputParser()
)

#prompt = "Quais são as DEFINIÇÕES estabelecidas no Ato 9523 de 27 de outubro de 2021?"
#rag.invoke(prompt)

# PASSO 9 - INTERFACE STREAMLIT Para fazer qualquer consulta com os passos encadeados, basta utilizar a função invoke.

# Interface Streamlit para o Chatbot
st.title("🛰️⚖️📘 Chatbot RAG sobre Requisitos Técnicos de Satélites - Ato SOR Anatel 9523/2021 ")
st.markdown("Faça perguntas e interaja com o Ato SOR 9523/2021 da Anatel para saber mais.")

pergunta = st.text_input("Digite sua pergunta:")

if st.button("Perguntar") and pergunta:
    with st.spinner("Consultando..."):
        resposta = rag.invoke(pergunta)
        st.markdown("### Resposta:")
        st.write(resposta)

#desabilitando modo debug para melhorar performance
#streamlit run app.py --server.runOnSave false

