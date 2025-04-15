"""
Código para criação de chatbot usando banco de dados vetorial (RAG) e LLM para interação com documento PDF
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv

# Carregar variáveis do ambiente
load_dotenv()

# Carregar chaves do Streamlit secrets
LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
HF_TOKEN = st.secrets["HF_TOKEN"]

# Função com cache para carregar base vetorial
@st.cache_resource(show_spinner="Carregando base vetorial...")
def carregar_base():
    arquivo = "Ato_Requisitos_Tecnicos_Satelites.pdf"
    dados = PyPDFLoader(arquivo).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    textos = text_splitter.split_documents(dados)
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.from_documents(textos, embedding_engine)
    return vector_db

vector_db = carregar_base()
retriever = vector_db.as_retriever()
n_documentos = 15

def format_docs(documentos):
    return "\n\n".join(documento.page_content for documento in documentos)

# Monta pipeline RAG com prompt, LLM e busca vetorial
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

rag = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Interface Streamlit
st.title("🛰️⚖️📘 Chatbot RAG sobre Requisitos Técnicos de Satélites - Ato SOR Anatel 9523/2021")
st.markdown("Faça perguntas e interaja com o Ato SOR 9523/2021 da Anatel para saber mais.")

if "historico" not in st.session_state:
    st.session_state.historico = []

if "input_pergunta" not in st.session_state:
    st.session_state.input_pergunta = ""

st.markdown("### Histórico da Conversa:")
for autor, msg in st.session_state.historico:
    st.markdown(f"**{autor}:** {msg}")

pergunta = st.text_input("Digite sua pergunta:", key="input_pergunta")

if st.button("Perguntar"):
    if pergunta:
        with st.spinner("Consultando..."):
            try:
                resposta = rag.invoke(pergunta)
            except Exception as e:
                resposta = f"Erro ao processar sua pergunta: {str(e)}"
            st.session_state.historico.append(("Você", pergunta))
            st.session_state.historico.append(("🤖 Sat-Bot", resposta))
        st.session_state.input_pergunta = ""
        st.rerun()
