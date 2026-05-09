import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def is_streamlit_context():
    """Check if we're running in Streamlit context"""
    try:
        import streamlit as st
        return hasattr(st, '_is_running_with_streamlit') or 'streamlit' in str(type(st))
    except:
        return False

def show_streamlit_warning(message):
    """Show warning only if in Streamlit context"""
    if is_streamlit_context():
        try:
            import streamlit as st
            st.warning(message)
        except:
            pass

def show_streamlit_error(message):
    """Show error only if in Streamlit context"""
    if is_streamlit_context():
        try:
            import streamlit as st
            st.error(message)
            for suggestion in [
                "• Scanned PDFs (image-based) - try using OCR or text-based PDFs",
                "• Password-protected files", 
                "• Corrupted files",
                "• Files with only images/graphics"
            ]:
                st.write(suggestion)
        except:
            pass

def load_specific_resumes(file_paths):
    """
    Loads specific PDF, DOCX, and TXT files from provided file paths.
    This function only processes the files that were just uploaded.
    
    Args:
        file_paths (list): List of file paths to process
        
    Returns:
        list: List of loaded documents with metadata
    """
    documents = []
    
    if not file_paths:
        print("No file paths provided")
        return documents
    
    print(f"Processing {len(file_paths)} uploaded files...")
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        # Check if file exists and has content
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue
            
        try:
            print(f"Processing uploaded file: {filename}")
            
            if filename.lower().endswith(".pdf"):
                # Load PDF with error handling
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Validate PDF content
                valid_docs = []
                for doc in docs:
                    content = doc.page_content.strip() if doc.page_content else ""
                    if content and len(content) > 10:  # At least 10 characters
                        doc.metadata["filename"] = filename
                        doc.metadata["file_type"] = "pdf"
                        doc.metadata["source"] = file_path
                        valid_docs.append(doc)
                        print(f"  - Page {len(valid_docs)}: {len(content)} characters")
                    else:
                        print(f"  - Skipped empty/short page in {filename}")
                
                if valid_docs:
                    documents.extend(valid_docs)
                    print(f"  ✅ Loaded {len(valid_docs)} pages from {filename}")
                else:
                    print(f"  ❌ PDF is not readable: {filename}")
                    show_streamlit_warning(f"PDF is not readable: {filename}. Please upload another PDF.")
                    
            elif filename.lower().endswith(".docx"):
                # Load DOCX with error handling
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                
                # Validate DOCX content
                valid_docs = []
                for doc in docs:
                    content = doc.page_content.strip() if doc.page_content else ""
                    if content and len(content) > 10:
                        doc.metadata["filename"] = filename
                        doc.metadata["file_type"] = "docx"
                        doc.metadata["source"] = file_path
                        valid_docs.append(doc)
                        print(f"  - Content: {len(content)} characters")
                    else:
                        print(f"  - Empty content in {filename}")
                
                if valid_docs:
                    documents.extend(valid_docs)
                    print(f"  ✅ Loaded {len(valid_docs)} documents from {filename}")
                else:
                    print(f"  ❌ No readable content found in {filename}")
                    show_streamlit_warning(f"⚠️ {filename}: No readable content found.")
                        
            elif filename.lower().endswith(".txt"):
                # Load TXT files
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    valid_docs = []
                    for doc in docs:
                        content = doc.page_content.strip() if doc.page_content else ""
                        if content and len(content) > 10:
                            doc.metadata["filename"] = filename
                            doc.metadata["file_type"] = "txt"
                            doc.metadata["source"] = file_path
                            valid_docs.append(doc)
                            print(f"  - Content: {len(content)} characters")
                        else:
                            print(f"  - Empty content in {filename}")
                    
                    if valid_docs:
                        documents.extend(valid_docs)
                        print(f"  ✅ Loaded {len(valid_docs)} text documents from {filename}")
                    else:
                        print(f"  ❌ No readable content found in {filename}")
                        
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        loader = TextLoader(file_path, encoding='latin-1')
                        docs = loader.load()
                        # Process same as above...
                        print(f"  ✅ Loaded {filename} with latin-1 encoding")
                    except Exception as e2:
                        print(f"  ❌ Could not read {filename} with any encoding: {e2}")
                        
        except Exception as e:
            error_msg = f"Error loading {filename}: {str(e)}"
            print(error_msg)
            show_streamlit_warning(f"Could not process {filename}: {str(e)}")
    
    print(f"Total documents loaded from uploaded files: {len(documents)}")
    return documents

def load_resumes(data_dir):
    """
    Loads PDF, DOCX, and TXT files from the specified directory with comprehensive error handling.
    
    Args:
        data_dir (str): Directory containing resume files
        
    Returns:
        list: List of loaded documents with metadata
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist")
        return documents
    
    # Get list of supported files
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    if not files:
        print(f"No supported files (PDF, DOCX, TXT) found in {data_dir}")
        return documents
    
    print(f"Found {len(files)} files to process: {files}")
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        
        # Check file size (skip empty files)
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue
            
        try:
            print(f"Processing: {filename}")
            
            if filename.lower().endswith(".pdf"):
                # Load PDF with error handling
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Validate PDF content
                valid_docs = []
                for doc in docs:
                    # Check if document has readable content
                    content = doc.page_content.strip() if doc.page_content else ""
                    if content and len(content) > 10:  # At least 10 characters
                        doc.metadata["filename"] = filename
                        doc.metadata["file_type"] = "pdf"
                        doc.metadata["source"] = file_path
                        valid_docs.append(doc)
                        print(f"  - Page {len(valid_docs)}: {len(content)} characters")
                    else:
                        print(f"  - Skipped empty/short page in {filename}")
                
                if valid_docs:
                    documents.extend(valid_docs)
                    print(f"  ✅ Loaded {len(valid_docs)} pages from {filename}")
                else:
                    print(f"  ❌ PDF is not readable: {filename}")
                    show_streamlit_warning(f"PDF is not readable: {filename}. Please upload another PDF.")
                    
            elif filename.lower().endswith(".docx"):
                # Load DOCX with error handling
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                
                # Validate DOCX content
                valid_docs = []
                for doc in docs:
                    content = doc.page_content.strip() if doc.page_content else ""
                    if content and len(content) > 10:
                        doc.metadata["filename"] = filename
                        doc.metadata["file_type"] = "docx"
                        doc.metadata["source"] = file_path
                        valid_docs.append(doc)
                        print(f"  - Content: {len(content)} characters")
                    else:
                        print(f"  - Empty content in {filename}")
                
                if valid_docs:
                    documents.extend(valid_docs)
                    print(f"  ✅ Loaded {len(valid_docs)} documents from {filename}")
                else:
                    print(f"  ❌ No readable content found in {filename}")
                    show_streamlit_warning(f"⚠️ {filename}: No readable content found.")
                        
            elif filename.lower().endswith(".txt"):
                # Load TXT files
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    valid_docs = []
                    for doc in docs:
                        content = doc.page_content.strip() if doc.page_content else ""
                        if content and len(content) > 10:
                            doc.metadata["filename"] = filename
                            doc.metadata["file_type"] = "txt"
                            doc.metadata["source"] = file_path
                            valid_docs.append(doc)
                            print(f"  - Content: {len(content)} characters")
                        else:
                            print(f"  - Empty content in {filename}")
                    
                    if valid_docs:
                        documents.extend(valid_docs)
                        print(f"  ✅ Loaded {len(valid_docs)} text documents from {filename}")
                    else:
                        print(f"  ❌ No readable content found in {filename}")
                        
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        loader = TextLoader(file_path, encoding='latin-1')
                        docs = loader.load()
                        # Process same as above...
                        print(f"  ✅ Loaded {filename} with latin-1 encoding")
                    except Exception as e2:
                        print(f"  ❌ Could not read {filename} with any encoding: {e2}")
                        
        except Exception as e:
            error_msg = f"Error loading {filename}: {str(e)}"
            print(error_msg)
            show_streamlit_warning(f"Could not process {filename}: {str(e)}")
    
    print(f"Total documents loaded: {len(documents)}")
    
    # Provide helpful feedback if no documents were loaded
    if not documents and files:
        error_msg = "No suitable candidates found. No readable content in any uploaded files."
        suggestions = [
            "• Scanned PDFs (image-based) - try using OCR or text-based PDFs",
            "• Password-protected files",
            "• Corrupted files",
            "• Files with only images/graphics"
        ]
        print(error_msg)
        for suggestion in suggestions:
            print(suggestion)
            
        show_streamlit_error(error_msg)
    
    return documents

def split_documents(documents):
    """
    Splits documents into chunks with comprehensive validation and metadata.
    
    Args:
        documents (list): List of Document objects
        
    Returns:
        list: List of document chunks with metadata
    """
    if not documents:
        print("No documents provided for splitting")
        return []
    
    print(f"Splitting {len(documents)} documents into chunks...")
    
    # Configure text splitter with optimal settings for resumes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,           # Reasonable chunk size for resume content
        chunk_overlap=100,         # Overlap to maintain context
        length_function=len,       # Use character count
        add_start_index=True,      # Track position in original document
        separators=["\n\n", "\n", ". ", " ", ""]  # Resume-friendly separators
    )
    
    try:
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            print("No chunks created from documents")
            return []
        
        print(f"Created {len(chunks)} initial chunks")
        
        # Filter out empty or very short chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            # Validate chunk content
            if chunk.page_content and len(chunk.page_content.strip()) >= 50:  # Minimum 50 characters
                # Add enhanced metadata
                chunk.metadata["chunk_id"] = len(valid_chunks)
                chunk.metadata["chunk_length"] = len(chunk.page_content)
                
                # Ensure filename is present
                if "filename" not in chunk.metadata:
                    if "source" in chunk.metadata:
                        chunk.metadata["filename"] = os.path.basename(chunk.metadata["source"])
                    else:
                        chunk.metadata["filename"] = "Unknown"
                
                # Add content preview for debugging
                preview = chunk.page_content[:100].replace('\n', ' ')
                chunk.metadata["content_preview"] = preview
                
                valid_chunks.append(chunk)
                print(f"  Chunk {len(valid_chunks)}: {len(chunk.page_content)} chars from {chunk.metadata.get('filename', 'Unknown')}")
            else:
                print(f"  Skipped short chunk: {len(chunk.page_content) if chunk.page_content else 0} chars")
        
        print(f"Final valid chunks: {len(valid_chunks)}")
        return valid_chunks
        
    except Exception as e:
        error_msg = f"Error splitting documents: {str(e)}"
        print(error_msg)
        show_streamlit_error(error_msg)
        return []

def validate_documents(documents):
    """
    Validates that documents contain readable content.
    
    Args:
        documents (list): List of Document objects
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not documents:
        return False, "No documents provided"
    
    total_content = 0
    valid_docs = 0
    
    for doc in documents:
        if doc.page_content and doc.page_content.strip():
            valid_docs += 1
            total_content += len(doc.page_content.strip())
    
    if valid_docs == 0:
        return False, "No documents contain readable text"
    
    if total_content < 100:  # Less than 100 characters total
        return False, f"Documents contain very little text ({total_content} characters)"
    
    return True, f"Valid: {valid_docs} documents with {total_content} characters"

if __name__ == "__main__":
    # Test loading and splitting
    print("Testing document loading and splitting...")
    
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")
    
    # Load documents
    docs = load_resumes("data")
    
    if docs:
        # Validate documents
        is_valid, message = validate_documents(docs)
        print(f"Validation: {message}")
        
        if is_valid:
            # Split into chunks
            chunks = split_documents(docs)
            print(f"\n✅ Successfully processed:")
            print(f"   - {len(docs)} documents")
            print(f"   - {len(chunks)} chunks")
            
            # Show sample chunk
            if chunks:
                sample = chunks[0]
                print(f"\nSample chunk from {sample.metadata.get('filename', 'Unknown')}:")
                print(f"Content preview: {sample.page_content[:200]}...")
        else:
            print(f"❌ Document validation failed: {message}")
    else:
        print("❌ No documents found in data/ folder")
        print("Please add PDF or DOCX resume files to the data/ directory")
