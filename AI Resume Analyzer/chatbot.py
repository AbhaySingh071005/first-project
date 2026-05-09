import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_chatbot_chain(vector_store):
    """
    Creates a RAG chain for the chatbot.
    """
    if not vector_store:
        return None

    # LLM Setup (Groq is used as default)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API_KEY_MISSING"
        
    llm = ChatGroq(
        temperature=0.1,
        model_name="llama-3.1-8b-instant",  # Updated to supported model
        groq_api_key=api_key
    )

    # Prompt Template
    template = """
    SYSTEM:
    You are an AI HR assistant. Analyze resumes and answer based only on given context.
    If the answer is not in the context, politely state that you don't have that information.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join([f"--- {doc.metadata.get('filename')} ---\n{doc.page_content}" for doc in docs])

    # Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def get_chat_response_stream(vector_store, question):
    """
    Generates a streaming response from the chatbot.
    """
    chain = get_chatbot_chain(vector_store)
    
    if chain == "API_KEY_MISSING":
        yield "Please provide a valid GROQ_API_KEY in the .env file."
        return
        
    if not chain:
        yield "Vector store not initialized. Please upload and process resumes first."
        return

    for chunk in chain.stream(question):
        yield chunk
