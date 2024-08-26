import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css, bot_template, user_template
import os


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY" # your google api key here


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever= vectorstore.as_retriever(),
        memory=memory,

    )
    return conversation_chain


def handle_user_input(user_question):
    response =  st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat App', page_icon=':speech_balloon:')

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat App :speech_balloon:')
    user_qustion = st.text_input('Welcome to the Chat App! Feel free to chat with me :smiley:')
    if user_qustion:
        handle_user_input(user_qustion)


    st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human") , unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader('Upload your documents here', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vectorstore
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation


if __name__ == '__main__':
    main()