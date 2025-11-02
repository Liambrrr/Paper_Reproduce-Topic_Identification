"""
Unified Metrics Calculation Utilities
Consolidates all metric calculation functions to avoid code duplication
"""
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from itertools import combinations


def rbo_score(list1, list2, p=0.9):
    """
    Calculate Rank-Biased Overlap (RBO) between two ranked lists
    
    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Weight parameter (default: 0.9)
        
    Returns:
        float: RBO score between 0 and 1
    """
    if not list1 or not list2:
        return 0.0
    
    k = min(len(list1), len(list2))
    if k == 0:
        return 0.0
    
    overlap = 0.0
    for d in range(1, k + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap += (len(set1 & set2) / d) * (p ** (d - 1))
    
    rbo = ((1 - p) / p) * overlap
    return min(1.0, rbo)


def calculate_coherence_cv_bertopic(texts, topics_list, topic_model, top_n=10):
    """
    Calculate C_v coherence score for BERTopic models
    
    Args:
        texts: List of text documents
        topics_list: List of topic assignments
        topic_model: Trained BERTopic model
        top_n: Number of top words to use (default: 10)
        
    Returns:
        float: Coherence score
    """
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts]
    
    # Build dictionary from texts
    dictionary = Dictionary(tokenized_texts)
    
    # Get all vocabulary from dictionary
    vocab = set(dictionary.token2id.keys())
    
    # Extract topic words
    topic_words = []
    unique_topics = sorted(set(topics_list))
    if -1 in unique_topics:
        unique_topics.remove(-1)
    
    for topic_id in unique_topics:
        try:
            words = topic_model.get_topic(topic_id)
            if words and len(words) > 0:
                # Extract word strings from tuples
                word_list = []
                for item in words[:top_n]:
                    if isinstance(item, tuple) and len(item) >= 1:
                        word = item[0]
                    else:
                        word = item
                    
                    # Ensure it's a string and in vocabulary
                    if isinstance(word, str) and len(word) > 0 and word in vocab:
                        word_list.append(word)
                
                # Only add if we have at least 2 words (minimum for coherence)
                if len(word_list) >= 2:
                    topic_words.append(word_list)
        except Exception as e:
            print(f"Warning: Could not get words for topic {topic_id}: {e}")
            continue
    
    # Need at least 2 topics with at least 2 words each
    if not topic_words or len(topic_words) < 2:
        return 0.0
    
    try:
        # Use Gensim's CoherenceModel with c_v metric
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # Avoid multiprocessing issues
        )
        coherence_score = coherence_model.get_coherence()
        
        # Sanity check
        if coherence_score is None or np.isnan(coherence_score):
            return 0.0
        
        return coherence_score
        
    except Exception as e:
        print(f"Warning: Could not calculate coherence: {e}")
        print(f"  Number of topics: {len(topic_words)}")
        print(f"  Topic words sample: {topic_words[:2] if len(topic_words) >= 2 else topic_words}")
        print(f"  Dictionary size: {len(dictionary)}")
        print(f"  Number of texts: {len(tokenized_texts)}")
        
        # Detailed debugging for first failed topic
        if topic_words:
            first_topic = topic_words[0]
            print(f"  First topic words: {first_topic}")
            in_dict = [word for word in first_topic if word in dictionary.token2id]
            print(f"  Words in dictionary: {in_dict}")
        
        return 0.0


def calculate_coherence_cv_traditional(texts, topic_words, top_n=10):
    """
    Calculate C_v coherence score for traditional models (LDA, LSI, NMF)
    
    Args:
        texts: List of text documents
        topic_words: List of lists of topic words
        top_n: Number of top words to use (default: 10)
        
    Returns:
        float: Coherence score
    """
    tokenized_texts = [text.split() for text in texts]
    dictionary = Dictionary(tokenized_texts)
    
    # Ensure topic_words are lists of words
    topic_words_filtered = []
    for words in topic_words:
        if words:
            topic_words_filtered.append(words[:top_n])
    
    if not topic_words_filtered:
        return 0.0
    
    try:
        coherence_model = CoherenceModel(
            topics=topic_words_filtered,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    except Exception as e:
        print(f"Warning: Could not calculate coherence: {e}")
        return 0.0


def calculate_irbo(topic_model, top_n=10, p=0.9):
    """
    Calculate IRBO (Inverted Rank-Biased Overlap) for topic diversity
    
    Args:
        topic_model: Trained BERTopic model
        top_n: Number of top words to use (default: 10)
        p: RBO weight parameter (default: 0.9)
        
    Returns:
        float: IRBO score (1 - mean RBO similarity)
    """
    topic_info = topic_model.get_topic_info()
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    if len(topic_ids) < 2:
        return 1.0
    
    topic_words_lists = []
    for topic_id in topic_ids:
        words = topic_model.get_topic(topic_id)
        if words:
            topic_words_lists.append([word for word, _ in words[:top_n]])
    
    similarities = []
    for list1, list2 in combinations(topic_words_lists, 2):
        sim = rbo_score(list1, list2, p=p)
        similarities.append(sim)
    
    if not similarities:
        return 1.0
    
    mean_similarity = np.mean(similarities)
    irbo = 1 - mean_similarity
    
    return irbo


def calculate_irbo_traditional(topic_words, top_n=10, p=0.9):
    """
    Calculate IRBO (Inverted Rank-Biased Overlap) for traditional models
    
    Args:
        topic_words: List of lists of topic words (each inner list is top words for a topic)
        top_n: Number of top words to use (default: 10)
        p: RBO weight parameter (default: 0.9)
        
    Returns:
        float: IRBO score (1 - mean RBO similarity)
    """
    if not topic_words or len(topic_words) < 2:
        return 1.0
    
    # Ensure we only use top_n words from each topic
    topic_words_lists = []
    for words in topic_words:
        if words and len(words) > 0:
            topic_words_lists.append(words[:top_n])
    
    if len(topic_words_lists) < 2:
        return 1.0
    
    # Calculate RBO similarity for all pairs of topics
    similarities = []
    for list1, list2 in combinations(topic_words_lists, 2):
        sim = rbo_score(list1, list2, p=p)
        similarities.append(sim)
    
    if not similarities:
        return 1.0
    
    # IRBO = 1 - mean similarity (higher IRBO = more diverse topics)
    mean_similarity = np.mean(similarities)
    irbo = 1 - mean_similarity
    
    return irbo