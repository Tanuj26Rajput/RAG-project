from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pdfplumber
import streamlit as st
import tempfile
import os

os.environ["HUGGINGFACEHUB_API_KEY"] = "your_api_key_here"

def load_docs(file_path):
    documents = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                docs = Document(
                    page_content=text,
                    metadata = {"page": page_number + 1}
                )
                documents.append(docs)
    return documents

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def building_chain(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunked_docs = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name = "intfloat/e5-small-v2",
        encode_kwargs = {"normalize_embeddings": True}
    )

    vector_store = FAISS.from_documents(
        documents=chunked_docs,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 4}
    )

    llm = HuggingFaceEndpoint(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        task = "text-generation",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )

    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""
            You are a helpful academic assistant. Use only the provided CONTEXT to answer the QUESTION.
            If the context is not enough, reply with "I don't know."

            CONTEXT:
            {context}

            QUESTION: {question}
            """,
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    final_chain = RunnableSequence(parallel_chain | prompt | llm | parser)
    return final_chain

st.set_page_config(page_title="PDF QNA", layout="centered")
st.title("📄📚 Chat With Your PDFs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None

uploaded_file = st.file_uploader("Upload your pdf", type="pdf")

if uploaded_file:
    with st.spinner("Reading..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        docu = load_docs(tmp_path)
        chain = building_chain(docu)
        st.session_state.chain = chain

question = st.text_input("Enter your question")

if question and st.session_state.chain:
    with st.spinner("Thinking..."):
        answer = st.session_state.chain.invoke(question)
        st.session_state.chat_history.append((question, answer.strip()))

if st.session_state.chat_history:
    st.markdown("### Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"You: {q}")
        st.markdown(f"Assistant: {a}")