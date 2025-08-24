# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# LangChain core components
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Anthropic LLM integration
from langchain_anthropic import ChatAnthropic

# Gradio for interface
import gradio as gr

# for file downloads
import wget


def load_document(filename='companyPolicies.txt'):
    """
    Function to load document from file
    Args:
        filename (str): Path to the document file
    Returns:
        documents: Loaded documents from the file
    """
    try:
        # Read the file content first to verify it exists
        with open(filename, 'r') as file:
            contents = file.read()
            print(f"Document loaded successfully. Length: {len(contents)} characters")
        
        # Load using LangChain TextLoader
        loader = TextLoader(filename)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None


def apply_text_splitting(documents, chunk_size=1000, chunk_overlap=0):
    """
    Function to apply text splitting technique
    Args:
        documents: Documents to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
    Returns:
        texts: Split text chunks
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    # print(f"Document split into {len(texts)} chunks")
    return texts


def define_embeddings():
    """
    Function to define embeddings model
    Returns:
        embeddings: HuggingFace embeddings model
    """
    embeddings = HuggingFaceEmbeddings()
    print("Embeddings model initialized")
    return embeddings


def configure_vector_db(texts, embeddings):
    """
    Function to configure vector database using ChromaDB to store embeddings
    Args:
        texts: Text chunks to embed
        embeddings: Embeddings model
    Returns:
        docsearch: ChromaDB vector store
    """
    docsearch = Chroma.from_documents(texts, embeddings) # add comma, .persist_directory="./chroma_db" if needed, currently its created in ram on run time.
    print("Vector database configured and documents ingested")
    return docsearch


def create_retriever(docsearch):
    """
    Function to create retriever for document segments
    Args:
        docsearch: Vector database
    Returns:
        retriever: Document retriever
    """
    retriever = docsearch.as_retriever()
    print("Retriever created")
    return retriever


def initialize_llm():
    """
    Function to initialize the LLM model
    Returns:
        llm: Initialized Anthropic Claude model
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",  # Current available model
        temperature=0.5,                      # Controls creativity
        max_tokens=256                        # Controls output length
    )
    print("LLM model initialized")
    return llm


def qa(question, chat_history=None):
    """
    Function for question-answering using LLM
    Args:
        question (str): User question
        chat_history: Previous conversation history
    Returns:
        answer (str): Generated answer
        updated_history: Updated chat history
    """
    global qa_chain
    
    if chat_history is None:
        chat_history = []
    
    try:
        # Format chat history for the chain
        formatted_history = []
        for human_msg, ai_msg in chat_history:
            formatted_history.append((human_msg, ai_msg))
        
        # Get response from the chain
        result = qa_chain({"question": question, "chat_history": formatted_history})
        answer = result["answer"]
        
        # Update chat history
        chat_history.append((question, answer))
        
        return answer, chat_history
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        chat_history.append((question, error_msg))
        return error_msg, chat_history


def setup_gradio_interface():
    """
    Function to setup Gradio interface
    Returns:
        interface: Gradio interface object
    """
    def chat_interface(message, history):
        """
        Chat interface function for Gradio
        """
        answer, updated_history = qa(message, history)
        return answer
    
    # Create Gradio ChatInterface
    interface = gr.ChatInterface(
        fn=chat_interface,
        title="RAG Private Document Q&A",
        description="Ask questions about the company policies document. The system will retrieve relevant information and provide answers.",
        examples=[
            "Can you summarize the document for me?",
            "What is the mobile policy?",
            "What are the working hours?",
            "Tell me about the leave policy"
        ]
    )
    
    return interface


def main():
    """
    Main function to orchestrate the RAG pipeline
    """
    global qa_chain
    
    print("Starting RAG Private Document Q&A System...")
    
    # Step 1: Load document
    documents = load_document('companyPolicies.txt')
    if documents is None:
        print("Failed to load document. Exiting...")
        return
    
    # Step 2: Apply text splitting
    texts = apply_text_splitting(documents)
    
    # Step 3: Define embeddings
    embeddings = define_embeddings()
    
    # Step 4: Configure vector database
    docsearch = configure_vector_db(texts, embeddings)
    
    # Step 5: Create retriever
    retriever = create_retriever(docsearch)
    
    # Step 6: Initialize LLM
    llm = initialize_llm()
    
    # Step 7: Create QA chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )
    
    print("RAG system initialized successfully!")
    
    # Step 8: Setup and launch Gradio interface
    interface = setup_gradio_interface()
    
    print("Launching Gradio interface...")
    interface.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
