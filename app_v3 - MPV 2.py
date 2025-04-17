
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

if "historico" not in st.session_state:
    st.session_state.historico = []

@st.cache_resource(show_spinner="Carregando base vetorial...")
def carregar_base():
    dados = PyPDFLoader("Ato_Requisitos_Tecnicos_Satelites.pdf").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    textos = splitter.split_documents(dados)
    embeddings = HuggingFaceEmbeddings("sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(textos, embeddings)

retriever = carregar_base().as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def obter_contexto(pergunta):
    docs = retriever.invoke(pergunta)
    return format_docs(docs)

def obter_historico(n=4):
    ultimas_interacoes = st.session_state.historico[-(n*2):]
    return "\n".join(f"{autor}: {msg}" for autor, msg in ultimas_interacoes)

prompt = PromptTemplate(
    input_variables=["contexto", "historico", "pergunta"],
    template="""
    Você é um assistente especialista no Ato SOR 9.523/2021 da Anatel sobre requisitos técnicos para satélites.
    Utilize o conteúdo do Ato ("contexto") para responder educadamente.
    Não invente respostas; se não souber, informe educadamente.

    Contexto:
    {contexto}

    Histórico recente:
    {historico}

    Pergunta atual:
    {pergunta}

    Resposta:
    """
)

llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0)

rag = (
    {
        "pergunta": RunnablePassthrough(),
        "contexto": RunnableLambda(obter_contexto),
        "historico": RunnableLambda(lambda _: obter_historico())
    }
    | prompt
    | llm
    | StrOutputParser()
)

st.title("🛰️⚖️📘 Sat-bot Regulatório - Ato SOR 9.523/2021")
st.markdown("Interaja com o chatbot especializado no Ato da Anatel.")

with st.container(height=400, border=True):
    st.markdown("### Histórico")
    for autor, mensagem in st.session_state.historico:
        st.markdown(f"**{autor}:** {mensagem}")

st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 10;
    }
    .st-emotion-cache-uf99v8 { padding-bottom: 100px; }
</style>
""", unsafe_allow_html=True)

with st.form("form_pergunta", clear_on_submit=True):
    pergunta = st.text_input("Digite sua pergunta:")
    enviar = st.form_submit_button("Perguntar")

if enviar and pergunta:
    with st.spinner("Consultando..."):
        try:
            resposta = rag.invoke(pergunta)
        except Exception as e:
            resposta = f"Erro: {str(e)}"
        st.session_state.historico.extend([("Você", pergunta), ("🤖 Sat-Bot", resposta)])
    st.rerun()

if st.button("Limpar Histórico"):
    st.session_state.historico = []
    st.rerun()
