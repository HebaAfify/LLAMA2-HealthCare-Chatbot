# Import Libraries

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message

# Load the PDF
loader = PyPDFLoader("mental_health_Document.pdf")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})

# Create vector database (vector store)
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create LLM
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128, "temperature": 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                              chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Create Streamlit app
st.title("HealthCare ChatBot 👩🏻‍⚕️")


def conversation_chat(query):
    # Append the answers to the history
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result['answer']))
    return result['answer']


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state['history'] = []
    if "generated" not in st.session_state:
        st.session_state['generated'] = ["Hii! Ask Heba anything"]
    if "past" not in st.session_state:
        st.session_state['past'] = ["Hey!"]


# Form the output
def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question: ", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='send')

            if submit_button and user_input:
                output = conversation_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    # Get the iteration of the conversations
    # Iterate through the conversation and extract the past and generated
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")


# Call Functions

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()
