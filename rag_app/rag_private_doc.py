# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn
# warnings.filterwarnings('ignore')

# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# from ibm_watsonx_ai.foundation_models import Model
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
# import wget

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

# for file downloads
import wget

filename = 'companyPolicies.txt'
# url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# # Use wget to download the file
# wget.download(url, out=filename)
# print('file downloaded')

# read the file
with open(filename, 'r') as file:
    # Read the contents of the file
    contents = file.read()
    # print(contents)

# `LangChain` is used to split the document and create chunks. It helps you divide a long story (document) into smaller parts, which are called `chunks`, so that it's easier to handle. 
# For the splitting process, the goal is to ensure that each segment is as extensive as if you were to count to a certain number of characters and meet the split separator. This certain number is called `chunk size`. 
# Let's set 1000 as the chunk size in this project. Though the chunk size is 1000, the splitting is happening randomly. This is an issue with LangChain. `CharacterTextSplitter` uses `\n\n` as the default split separator. 
# You can change it by adding the `separator` parameter in the `CharacterTextSplitter` function; for example, `separator="\n"`.

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))


embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested')

#####################################################################################
## LLM model construction
######################################################################################

# Anthropic LLM integration
from langchain_anthropic import ChatAnthropic  # For Opus, Haiku, Claude 2/3 models
from langchain.chains import RetrievalQA

# Initialize Anthropic model (using Claude 3.5 Sonnet which is currently available)
opus_llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # Current available model
    temperature=0.5,                      # Controls creativity
    max_tokens=256                        # Controls output length
)


def qa():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=opus_llm,
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=False
    )
    
    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit","exit","bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa({"question": query})
        
        print("Answer: ", result["answer"])


qa()



# ############
# # this block gyudes LLM if question is outside the document context
# ############
# prompt_template = """Use the information from the document to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

# {context}

# Question: {question}
# """

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# chain_type_kwargs = {"prompt": PROMPT}
# ############
# # you can remove this above block, but it helps to guide the LLM to not make up answers
# ############

# ############
# ## This block defines memory to retain the context of the conversation
# ############
# memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)



# # Build QA chain using Anthropic model
# qa = RetrievalQA.from_chain_type(
#     llm=opus_llm,
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(),
#     chain_type_kwargs=chain_type_kwargs, 
#     memory = memory, 
#     get_chat_history=lambda h : h, 
#     return_source_documents=False
# )

# # Run query
# query = "where is Australia?"#"Can you summarize the document for me?" # "what is mobile policy?"
# response = qa.invoke(query)
# print(response)
