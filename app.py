import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template   
from langchain.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
#import pinecone
from pinecone import Pinecone
#from langchain.vectorstores import Pinecone as lang_pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
import os




def create_index(index_name, dimension, docs, embeddings):
    
    # if index_name in pinecone.list_indexes():
    #     pinecone.delete_index(index_name)
    
    # pinecone.create_index(name=index_name, metric="cosine", dimension=dimension)
    
    #index = Pinecone.from_documents(documents = docs, embedding=embeddings, index_name = index_name)
    index = PineconeVectorStore.from_existing_index(index_name,embeddings)
    return index



def connect_db():
    load_dotenv()
    api_key = os.environ.get('PINECONE_API_KEY')
    Pinecone(
    environment="gcp-starter"
    )

def load_docs(filename):
    loader = PyPDFLoader(filename, extract_images=True)
    pages = loader.load()
    return pages

def get_pdf_text(pdf_docs):
    documents = load_docs(pdf_docs)
    return documents

    
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents) #to use across multiple documents rather than a string, use split_documents
    return docs


def get_vectorstore(text_chunks,DIMENSION):
    embeddings = OllamaEmbeddings()

    vectorstore = create_index('langchain-pdfqna', DIMENSION, docs = text_chunks, embeddings=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = Ollama(model="llama2")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":1}),
        memory=memory,
        return_source_documents = True
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    print(response['source_documents'])
    st.session_state.chat_history = response['chat_history']
    print(st.session_state['chat_history'])

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    DIMENSION = 4096
    connect_db()
    st.set_page_config(page_title="Scientific Document Q&A",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Scientific Document Q&A")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                temp_file = "./temp.pdf"
                with open(temp_file, "wb") as file:
                    file.write(pdf_docs[0].getvalue())
                    file_name = pdf_docs[0].name
                # get pdf text
                # documents = get_pdf_text(temp_file)
                # print("read pdf")

                # # get the text chunks
                # text_chunks = split_docs(documents)
                # print("got chunks")


                # # create vector store
                vectorstore = get_vectorstore(text_chunks="text_chunks", DIMENSION=DIMENSION)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                print("chatting ")


if __name__ == '__main__':
    main()
