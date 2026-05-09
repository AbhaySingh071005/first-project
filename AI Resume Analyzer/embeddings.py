import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Vector store configuration
VECTOR_DB_PATH = "vectorstore/faiss_index"

def get_embeddings():
    """
    Returns the Sentence Transformer embedding model with error handling.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model
    """
    try:
        print("Initializing embedding model...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print(f"✅ Embedding model loaded: {model_name}")
        return embeddings
        
    except Exception as e:
        error_msg = f"Failed to load embedding model: {str(e)}"
        print(f"❌ {error_msg}")
        if hasattr(st, 'error'):
            st.error(error_msg)
        raise Exception(error_msg)

def validate_chunks_for_embedding(chunks):
    """
    Validates chunks before creating embeddings to prevent IndexError.
    
    Args:
        chunks (list): List of document chunks
        
    Returns:
        tuple: (is_valid, error_message, valid_chunks)
    """
    if not chunks:
        return False, "No chunks provided", []
    
    if not isinstance(chunks, list):
        return False, "Chunks must be a list", []
    
    valid_chunks = []
    total_content_length = 0
    
    for i, chunk in enumerate(chunks):
        # Check if chunk is a Document object
        if not isinstance(chunk, Document):
            print(f"Warning: Chunk {i} is not a Document object: {type(chunk)}")
            continue
            
        # Check if chunk has content
        if not hasattr(chunk, 'page_content') or not chunk.page_content:
            print(f"Warning: Chunk {i} has no page_content")
            continue
            
        # Check content length
        content = chunk.page_content.strip()
        if len(content) < 10:  # Minimum content length
            print(f"Warning: Chunk {i} content too short: {len(content)} chars")
            continue
            
        valid_chunks.append(chunk)
        total_content_length += len(content)
        
    if not valid_chunks:
        return False, "No valid chunks with sufficient content", []
        
    if total_content_length < 50:
        return False, f"Total content too short: {total_content_length} characters", []
    
    print(f"Validation passed: {len(valid_chunks)} valid chunks, {total_content_length} total characters")
    return True, f"Valid: {len(valid_chunks)} chunks", valid_chunks

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks with comprehensive error handling.
    
    Args:
        chunks (list): List of Document objects with text content
        
    Returns:
        FAISS: Vector store object or None if creation fails
    """
    try:
        print(f"Creating vector store from {len(chunks) if chunks else 0} chunks...")
        
        # Validate input chunks
        is_valid, message, valid_chunks = validate_chunks_for_embedding(chunks)
        
        if not is_valid:
            error_msg = f"Chunk validation failed: {message}"
            print(f"❌ {error_msg}")
            if hasattr(st, 'error'):
                st.error(error_msg)
            return None
        
        print(f"Using {len(valid_chunks)} validated chunks")
        
        # Get embeddings model
        embeddings = get_embeddings()
        
        # Test embedding with a sample to catch issues early
        print("Testing embedding generation...")
        sample_text = valid_chunks[0].page_content[:500]  # Use first 500 chars as test
        test_embedding = embeddings.embed_query(sample_text)
        
        if not test_embedding or len(test_embedding) == 0:
            error_msg = "Embedding generation failed - empty embedding returned"
            print(f"❌ {error_msg}")
            if hasattr(st, 'error'):
                st.error(error_msg)
            return None
            
        print(f"✅ Test embedding successful: {len(test_embedding)} dimensions")
        
        # Create FAISS vector store
        print("Creating FAISS index...")
        vector_store = FAISS.from_documents(valid_chunks, embeddings)
        
        if vector_store is None:
            error_msg = "FAISS vector store creation returned None"
            print(f"❌ {error_msg}")
            if hasattr(st, 'error'):
                st.error(error_msg)
            return None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        
        # Save vector store
        print(f"Saving vector store to {VECTOR_DB_PATH}...")
        vector_store.save_local(VECTOR_DB_PATH)
        
        # Verify save was successful
        index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
        if not os.path.exists(index_file):
            error_msg = "Vector store save failed - index file not created"
            print(f"❌ {error_msg}")
            if hasattr(st, 'error'):
                st.error(error_msg)
            return None
        
        print(f"✅ Vector store created successfully with {len(valid_chunks)} documents")
        return vector_store
        
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        print(f"❌ {error_msg}")
        if hasattr(st, 'error'):
            st.error(error_msg)
        return None

def load_vector_store():
    """
    Loads the FAISS vector store from local storage with error handling.
    
    Returns:
        FAISS: Loaded vector store or None if loading fails
    """
    try:
        # Check if the index file exists
        index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
        pkl_file = os.path.join(VECTOR_DB_PATH, "index.pkl")
        
        if not os.path.exists(index_file):
            print(f"No existing vector store found at {VECTOR_DB_PATH}")
            return None
            
        if not os.path.exists(pkl_file):
            print(f"Vector store incomplete - missing pkl file at {VECTOR_DB_PATH}")
            return None
        
        print(f"Loading vector store from {VECTOR_DB_PATH}...")
        
        # Get embeddings model
        embeddings = get_embeddings()
        
        # Load the vector store
        vector_store = FAISS.load_local(
            VECTOR_DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        if vector_store is None:
            print("❌ Vector store loading returned None")
            return None
            
        # Test the loaded vector store
        try:
            # Try a simple similarity search to verify it works
            test_results = vector_store.similarity_search("test", k=1)
            print(f"✅ Vector store loaded successfully with {vector_store.index.ntotal} vectors")
            return vector_store
            
        except Exception as test_error:
            print(f"❌ Vector store loaded but failed test: {str(test_error)}")
            return None
            
    except Exception as e:
        error_msg = f"Error loading vector store: {str(e)}"
        print(f"❌ {error_msg}")
        # Don't show error in Streamlit for loading - it's expected on first run
        return None

def get_vector_store_info(vector_store):
    """
    Gets information about the vector store for debugging.
    
    Args:
        vector_store (FAISS): Vector store object
        
    Returns:
        dict: Information about the vector store
    """
    if vector_store is None:
        return {"status": "None", "count": 0}
    
    try:
        info = {
            "status": "loaded",
            "count": vector_store.index.ntotal,
            "dimension": vector_store.index.d,
        }
        return info
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # Test embeddings and vector store functionality
    print("Testing embeddings and vector store logic...")
    
    try:
        # Test embedding model
        embeddings = get_embeddings()
        test_text = "This is a test resume with Python programming skills."
        test_embedding = embeddings.embed_query(test_text)
        print(f"✅ Embedding test successful: {len(test_embedding)} dimensions")
        
        # Test with sample document
        from langchain_core.documents import Document
        
        sample_docs = [
            Document(
                page_content="John Doe is a software engineer with 5 years of Python experience.",
                metadata={"filename": "test_resume.pdf", "chunk_id": 0}
            ),
            Document(
                page_content="Skills include machine learning, data analysis, and web development.",
                metadata={"filename": "test_resume.pdf", "chunk_id": 1}
            )
        ]
        
        # Test vector store creation
        print("\nTesting vector store creation...")
        vector_store = create_vector_store(sample_docs)
        
        if vector_store:
            info = get_vector_store_info(vector_store)
            print(f"✅ Vector store test successful: {info}")
            
            # Test search
            results = vector_store.similarity_search("Python developer", k=1)
            print(f"✅ Search test successful: found {len(results)} results")
        else:
            print("❌ Vector store creation failed")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
