import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
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

# Prompt customizado orientando o chatbot a não alucinar e aprimorar o contexto.
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Você é um assistente especializado do Satbot regulatório focado em prover orientações específicas 
    sobre o Ato SOR 9.523/2021 da Anatel sobre requisitos técnicos para satélites. Utilize o conteúdo do Ato (fornecido no campo "contexto") para responder de forma educada e clara. 
    Se você não encontrar a resposta na documentação, **não invente**. Em vez disso, informe gentilmente que não encontrou a resposta no Ato e peça mais detalhes ao usuário.
    Importante: sempre que possível, inclua no final da resposta, entre parênteses, o número do item do Ato correspondente à resposta, por exemplo: (Item 6.2.1).

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

#Usa biblioteca Langchain p/ criar pipeline para chatbot com RAG
rag = (
    {
        "question": RunnablePassthrough(),  #pergunta do usuario é repassada sem alterações
        "context": retriever | format_docs # contexto é obtido do banco de dados vetorial e formatado
    }
    | prompt # prompt é enviado para o modelo de linguagem desejado 
    | llm # modelo de linguagem é chamado (LLM da OpenAI) e recebe o prompt
    | StrOutputParser() #converte a resposata do LLM em string
)

# Interface Streamlit
st.title("🛰️⚖️📘 'Sat-bot' Regulatório - Requisitos Técnicos de Satélites - Ato SOR Anatel 9523/2021")
st.markdown("Faça perguntas e interaja com o Ato SOR 9.523/2021 da Anatel para saber mais.")

#Inicializa o histórico
if "historico" not in st.session_state:
    st.session_state.historico = []

#Historico de conversas
historico_placeholder = st.container(height=500, border=True)
with historico_placeholder:
    st.markdown("### Histórico da Conversa:")
    for autor, msg in st.session_state.historico:
        st.markdown(f"**{autor}:** {msg}")

# Espaço vazio para ajustar layout
st.write("<br><br>", unsafe_allow_html=True)

# Caixa de entrada fixa na parte inferior usando CSS
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
    .st-emotion-cache-uf99v8 {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# Formulário fixado embaixo
with st.form("form_pergunta", clear_on_submit=True):
    pergunta = st.text_input("Digite sua pergunta:", key="input_pergunta")
    enviar = st.form_submit_button("Perguntar")
if enviar and pergunta:
    with st.spinner("Consultando..."):
        try:
            resposta = rag.invoke(pergunta)
        except Exception as e:
            resposta = f"Erro ao processar sua pergunta: {str(e)}"
        st.session_state.historico.append(("Você", pergunta))
        st.session_state.historico.append(("🤖 Sat-Bot", resposta))
    st.rerun()
    
# Botão para limpar histórico
if st.button("Limpar histórico"):
    st.session_state.historico = []
    st.rerun()
