import bs4
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Rock Climbing RAG", page_icon="🧗")
st.title("🧗 Rock Climbing Assistant")
st.write("Ask anything about rock climbing!")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return ChatOllama(model="llama3")

model = load_model()

# -----------------------------
# Load & Index Data
# -----------------------------
@st.cache_resource
def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = InMemoryVectorStore(embeddings)

    loader = WebBaseLoader(
        web_paths=("https://en.wikipedia.org/wiki/Rock_climbing",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(id="bodyContent")
        ),
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)
    vector_store.add_documents(splits)

    return vector_store

with st.spinner("Ładowanie danych i budowanie bazy wektorowej..."):
    vector_store = setup_vectorstore()

# -----------------------------
# Retrieval Function
# -----------------------------
def retrieve_context(query: str):
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return "\n\n".join(
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

# -----------------------------
# Chat Interface
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question:")

if st.button("Ask") and user_input:

    st.session_state.chat_history.append(("You", user_input))

    with st.spinner("🤖 Model is thinking..."):
        context = retrieve_context(user_input)

        prompt = f"""
        Answer the question using the context below.

        Question:
        {user_input}

        Context:
        {context}
        """

        response = model.invoke(
            [{"role": "user", "content": prompt}]
        )

    st.session_state.chat_history.append(("Assistant", response.content))

# -----------------------------
# Display Chat
# -----------------------------
for role, message in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Assistant:** {message}")