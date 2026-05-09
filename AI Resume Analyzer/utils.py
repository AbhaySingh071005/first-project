import os
import shutil
import re

def save_uploaded_file(uploaded_file, data_dir="data"):
    """
    Saves an uploaded file to the local data directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def clear_data_directory(data_dir="data", vector_dir="vectorstore"):
    """
    Clears the data and vectorstore directories.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
    os.makedirs(vector_dir)

def highlight_keywords(text, keywords):
    """
    Wraps keywords in markdown bold/color for highlighting.
    """
    if not keywords:
        return text
    
    # Sort keywords by length descending to avoid partial matches
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    for kw in sorted_keywords:
        if len(kw) < 3: continue # Skip very short common words
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f"**{kw}**", text)
        
    return text

def get_file_download_link(file_path):
    """
    Provides a way for users to download the original resume.
    """
    # In Streamlit, we usually use st.download_button
    with open(file_path, "rb") as f:
        return f.read()
