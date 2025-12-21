def get_relevant_context(question: str, vectorstore, k=8):
    """Intelligent retrieval with adaptive context window"""
    power_keywords = ['top', 'most', 'best', 'all', 'list', 'expensive', 
                      'cheapest', 'compare', 'ranking', 'every', 'entire',
                      'order', 'sorted', 'ranked', 'highest', 'lowest']
    
    # Adaptive retrieval based on query complexity
    # Increase retrieval for ranking/comparison queries
    if any(word in question.lower() for word in power_keywords):
        k = 30  # Increased from 20 to 30 for better coverage
    
    # For very specific ranking queries (top 10, top 5, etc.)
    import re
    top_n_match = re.search(r'top\s+(\d+)', question.lower())
    if top_n_match:
        requested_count = int(top_n_match.group(1))
        # Retrieve significantly more chunks to ensure we have enough items
        k = max(40, requested_count * 4)  # 4x multiplier instead of 2x
    
    docs = vectorstore.similarity_search(question, k=k)
    
    # Deduplicate and consolidate information by item name
    seen_items = {}
    unique_docs = []
    
    for doc in docs:
        content = doc.page_content
        # Extract potential item names (simple heuristic)
        lines = content.split('\n')
        item_key = lines[0][:50] if lines else content[:50]
        
        if item_key not in seen_items:
            seen_items[item_key] = doc
            unique_docs.append(doc)
    
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    
    return context, unique_docs, k
