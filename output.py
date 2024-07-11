import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# Initialize the OpenAI API (make sure you have your OpenAI API key set up)
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_api_key2 = st.secrets["secret_section"]["OPENAI_API_KEY"]
# Replace with your actual OpenAI API key
st.set_page_config(page_title="Questions and Answers from document ", layout="wide")

# Initialize session state for maintaining history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title("Ask Questions from your PDF Document")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    st.write(f"PDF loaded and split into {len(texts)} chunks.")

    embeddings = OpenAIEmbeddings(api_key=openai_api_key2)
    document_search = FAISS.from_texts(texts, embeddings)
    st.write("Document embeddings created and stored in FAISS index.")

    chain = load_qa_chain(OpenAI(api_key=openai_api_key2), chain_type="stuff")

    query = st.chat_input("Ask a question about the PDF:")
    if query:
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.session_state.history.append((query, answer))
        st.write("Answer:", answer)

# Display history
if st.session_state.history:
    st.write("### History")
    for i, (question, answer) in enumerate(st.session_state.history):
        st.write(f"**Q{i+1}:** {question}")
        st.write(f"**A{i+1}:** {answer}")
