import os
import re

def search_resumes(vector_store, job_description, top_k=10):
    """
    Performs semantic search on the vector store using the job description.
    Returns a list of unique candidates with their highest similarity scores.
    """
    if not vector_store:
        return []

    # FAISS relevance scores are typically 1 - distance. 
    # With normalized embeddings, cosine similarity is 1 - distance^2/2 or similar.
    # similarity_search_with_relevance_scores filters out negative scores by default.
    try:
        results = vector_store.similarity_search_with_relevance_scores(job_description, k=top_k*2)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        # Fallback to standard similarity search if relevance scores fail
        docs = vector_store.similarity_search(job_description, k=top_k*2)
        results = [(doc, 0.5) for doc in docs] # Assign a neutral score
    
    # Process results into a more usable format, grouping by candidate (filename)
    candidates = {}
    for doc, score in results:
        filename = doc.metadata.get("filename", "Unknown")
        if filename not in candidates or score > candidates[filename]["score"]:
            candidates[filename] = {
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            
    # Convert to list and sort by score descending
    sorted_candidates = sorted(
        [{"filename": k, **v} for k, v in candidates.items()],
        key=lambda x: x["score"],
        reverse=True
    )
    
    return sorted_candidates[:top_k]

def hybrid_search(vector_store, job_description, keywords=None, top_k=10):
    """
    Combines semantic search with keyword filtering/boosting.
    """
    semantic_results = search_resumes(vector_store, job_description, top_k=top_k * 2)
    
    if not keywords:
        return semantic_results[:top_k]
    
    # Simple keyword boosting: increase score if keywords are present in the content
    for result in semantic_results:
        content_lower = result["content"].lower()
        keyword_matches = 0
        for kw in keywords:
            if kw.lower() in content_lower:
                keyword_matches += 1
        
        # Boost score: +5% for each keyword match (just a simple heuristic)
        if keyword_matches > 0:
            result["score"] += (keyword_matches * 0.05)
            result["keyword_matches"] = keyword_matches
        else:
            result["keyword_matches"] = 0

    # Re-sort after boosting
    sorted_results = sorted(semantic_results, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]

def extract_keywords(text):
    """
    Basic helper to extract potential keywords from a JD (nouns/tech terms).
    For now, we'll keep it simple or let the user provide them.
    """
    # Just a placeholder for more complex NLP if needed
    return re.findall(r'\b\w{3,}\b', text.lower())
