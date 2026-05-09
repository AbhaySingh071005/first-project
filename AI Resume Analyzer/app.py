import streamlit as st
import os
from datetime import datetime
from loader import load_resumes, split_documents, load_specific_resumes
from embeddings import create_vector_store, load_vector_store
from retriever import search_resumes, hybrid_search, extract_keywords
from chatbot import get_chat_response_stream
from utils import save_uploaded_file, clear_data_directory, highlight_keywords
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(page_title="IntelliRecruit AI", page_icon="📄", layout="wide")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_resumes" not in st.session_state:
    st.session_state.uploaded_resumes = []

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Resumes", "Job Matching", "Chat with AI HR"])

# ----------------- PAGE 1: UPLOAD -----------------
if page == "Upload Resumes":
    st.title("📄 Upload & Process Resumes")
    st.info("Upload multiple PDF or DOCX resumes to build the semantic search index.")

    uploaded_files = st.file_uploader("Choose resume files", type=["pdf", "docx"], accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process & Index Resumes"):
            if uploaded_files:
                try:
                    with st.spinner("Processing documents..."):
                        # Set current upload time
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Clear existing data to ensure only new uploads are processed
                        clear_data_directory()
                        
                        # Save files locally with validation
                        saved_files = []
                        saved_file_paths = []
                        for file in uploaded_files:
                            saved_path = save_uploaded_file(file)
                            if saved_path:
                                saved_files.append(file.name)
                                saved_file_paths.append(saved_path)
                            else:
                                st.warning(f"Failed to save {file.name}")
                        
                        if not saved_files:
                            st.error("No files were successfully saved. Please try again.")
                            st.stop()
                        
                        # Load and validate ONLY the uploaded documents
                        documents = load_specific_resumes(saved_file_paths)
                        
                        if not documents:
                            st.error("❌ No suitable candidates found. No readable content in uploaded files.")
                            st.info("💡 **Common issues:**")
                            st.write("• Scanned PDFs (image-based) - try text-based PDFs")
                            st.write("• Password-protected files")
                            st.write("• Corrupted or empty files")
                            st.write("• Files containing only images/graphics")
                            st.stop()
                        
                        # Split documents into chunks
                        chunks = split_documents(documents)
                        
                        if not chunks:
                            st.error("❌ No text chunks could be created. Documents may be empty or corrupted.")
                            st.stop()
                        
                        # Create vector store with error handling
                        vector_store = create_vector_store(chunks)
                        
                        if vector_store is None:
                            st.error("❌ Failed to create vector store. Please check your documents and try again.")
                            st.stop()
                        
                        # Success - update session state
                        st.session_state.vector_store = vector_store
                        
                        # Update uploaded resumes list with file info
                        resume_info = []
                        for i, filename in enumerate(saved_files):
                            # Get document info for this file
                            file_docs = [doc for doc in documents if doc.metadata.get('filename') == filename]
                            file_chunks = [chunk for chunk in chunks if chunk.metadata.get('filename') == filename]
                            
                            resume_info.append({
                                'filename': filename,
                                'pages': len(file_docs),
                                'chunks': len(file_chunks),
                                'status': 'Successfully processed',
                                'upload_time': current_time
                            })
                        
                        # Update session state with new resumes (replace old list)
                        st.session_state.uploaded_resumes = resume_info
                    
                    # Show success message after spinner completes
                    st.success(f"✅ Successfully indexed {len(saved_files)} resumes ({len(chunks)} chunks)!")
                    
                    # Show processing summary
                    with st.expander("📊 Processing Summary"):
                        st.write(f"**Files processed:** {len(saved_files)}")
                        st.write(f"**Document pages:** {len(documents)}")
                        st.write(f"**Text chunks:** {len(chunks)}")
                        st.write(f"**Files:** {', '.join(saved_files)}")
                            
                except Exception as e:
                    st.error(f"❌ Error processing resumes: {str(e)}")
                    st.error("Please check your files and try again. Ensure PDFs contain readable text.")
                    st.stop()
            else:
                st.warning("Please upload at least one resume.")

    with col2:
        if st.button("Clear All Data", type="secondary"):
            try:
                clear_data_directory()
                st.session_state.vector_store = None
                st.session_state.chat_history = []
                st.session_state.uploaded_resumes = []  # Clear uploaded resumes list
                st.success("All data cleared.")
            except Exception as e:
                st.error(f"Error clearing data: {str(e)}")
    
    # Display uploaded resumes list ONLY after successful processing
    if st.session_state.uploaded_resumes:
        st.markdown("---")
        st.subheader("📋 Successfully Processed Resumes")
        
        for i, resume in enumerate(st.session_state.uploaded_resumes):
            with st.expander(f"📄 {resume['filename']}", expanded=False):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("Pages", resume['pages'])
                
                with col_info2:
                    st.metric("Text Chunks", resume['chunks'])
                
                with col_info3:
                    st.write(f"**Status:** {resume['status']}")
                    st.write(f"**Uploaded:** {resume['upload_time']}")
                
                # Show download button if file exists
                file_path = os.path.join("data", resume['filename'])
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {resume['filename']}",
                            data=f,
                            file_name=resume['filename'],
                            key=f"download_resume_{i}",
                            help="Download the original resume file"
                        )

# ----------------- PAGE 2: JOB MATCHING -----------------
elif page == "Job Matching":
    st.title("🔍 Job Description Matching")
    
    if not st.session_state.vector_store:
        st.warning("Please upload and process resumes first.")
    else:
        jd_input = st.text_area("Enter Job Description", placeholder="Looking for a Python Developer with experience in Machine Learning and FastAPI...", height=200)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_n = st.slider("Top Candidates", 1, 10, 5)
        
        if st.button("Find Best Matches"):
            if jd_input:
                with st.spinner("Searching..."):
                    keywords = extract_keywords(jd_input)
                    results = hybrid_search(st.session_state.vector_store, jd_input, keywords=keywords, top_k=top_n)
                    
                    if results:
                        st.subheader(f"Top {len(results)} Matching Candidates")
                        for i, res in enumerate(results):
                            with st.expander(f"#{i+1} - {res['filename']} (Score: {res['score']:.2f})"):
                                st.write("**Top Relevant Chunk:**")
                                highlighted_text = highlight_keywords(res['content'], keywords)
                                st.markdown(highlighted_text)
                                
                                # Download button for the file
                                file_path = os.path.join("data", res['filename'])
                                if os.path.exists(file_path):
                                    with open(file_path, "rb") as f:
                                        st.download_button(
                                            label=f"Download {res['filename']}",
                                            data=f,
                                            file_name=res['filename'],
                                            key=f"dl_{i}"
                                        )
                    else:
                        st.write("No matching candidates found.")
            else:
                st.warning("Please enter a job description.")

# ----------------- PAGE 3: CHATBOT -----------------
elif page == "Chat with AI HR":
    st.title("🤖 AI HR Assistant")
    
    if not st.session_state.vector_store:
        st.warning("Please upload and process resumes first.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me about the candidates..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in get_chat_response_stream(st.session_state.vector_store, prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
