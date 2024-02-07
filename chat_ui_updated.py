import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
# from langchain import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Use Streamlit's caching mechanism to load and cache the model and vector store
# @st.experimental_singleton
@st.cache_resource
def load_model_and_vectorstore():
    # Initialize the openai language model
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # Initialze the huggingface language model
    # llm=HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0,"max_length":64})


    # Load and prepare document
    document_path = "../data/Common_RFI_RFP_Questions.pdf"
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    
    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Persist the vectors locally on disk
    vectorstore.save_local("faiss_index_datamodel")    

    
    # Define your system instruction
    system_instruction = """As an AI assistant, you must answer the query from the user from the retrieved content,
    if no relevant information is available, answer the question by using your knowledge about the topic, and dont mention Based on the retrieved content or According to the retrieved content sentense in the answer. and answer dont start with A: or assistant: anything like this."""
    
    # Define your template with the system instruction
    template = (
        f"{system_instruction} "
        "Combine the chat history{chat_history} and follow up question into "
        "a standalone question to answer from the {context}. "
        "Follow up question: {question}"
    )

    prompt = PromptTemplate.from_template(template)
    
    # Initialize ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        #condense_question_prompt=prompt,
        combine_docs_chain_kwargs={'prompt': prompt},
        chain_type="stuff",
    )
    
    return chain

# Function to handle the chat
def handle_chat(prompt):
    # Access the global chat history stored in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Load model and vectorstore only once
    chain = load_model_and_vectorstore()
    
    # Process the query
    response = chain({"question": prompt, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((prompt, response["answer"]))
    
    return response['answer']

# Streamlit UI components
st.title("RFI/RFP Assistant")

# Initialize chat history UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the response from the model
    response = handle_chat(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
