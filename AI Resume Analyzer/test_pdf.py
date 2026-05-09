#!/usr/bin/env python3
"""Test script to debug PDF loading issues"""

import os
from langchain_community.document_loaders import PyPDFLoader

def test_pdf_loading():
    """Test PDF loading with detailed debugging"""
    
    pdf_file = "data/aarav resume.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"File not found: {pdf_file}")
        return
    
    file_size = os.path.getsize(pdf_file)
    print(f"File: {pdf_file}")
    print(f"Size: {file_size} bytes")
    
    try:
        # Test with PyPDFLoader
        print("\nTesting with PyPDFLoader...")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        
        print(f"Documents loaded: {len(docs)}")
        
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            print(f"Page {i+1}: {len(content)} characters")
            if content:
                print(f"First 200 chars: {content[:200]}")
            else:
                print("Empty content")
            print(f"Metadata: {doc.metadata}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error with PyPDFLoader: {e}")
    
    # Test with pypdf directly
    try:
        print("\nTesting with pypdf directly...")
        import pypdf
        
        with open(pdf_file, 'rb') as file:
            reader = pypdf.PdfReader(file)
            print(f"Pages: {len(reader.pages)}")
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                print(f"Page {i+1}: {len(text)} characters")
                if text.strip():
                    print(f"First 200 chars: {text[:200]}")
                else:
                    print("Empty or whitespace only")
                    
    except Exception as e:
        print(f"Error with pypdf: {e}")

if __name__ == "__main__":
    test_pdf_loading()