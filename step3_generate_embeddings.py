"""
Step 3: Generate Sentence Embeddings
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("="*60)
    print("Step 3: Generate Sentence Embeddings")
    print("="*60)
    
    # Load preprocessed groups
    print("\n[1/3] Loading preprocessed groups...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        groups = pickle.load(f)
    
    # Load embedding model
    print("\n[2/3] Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded")
    
    # Generate embeddings
    print("\n[3/3] Generating embeddings for all groups...")
    embeddings = {}
    
    for name, group_df in groups.items():
        print(f"\n  Generating embeddings for {name} ({len(group_df)} posts)...")
        texts = group_df['cleaned_text'].tolist()
        embeddings[name] = embedding_model.encode(texts, show_progress_bar=True)
        print(f"    Shape: {embeddings[name].shape}")
    
    # Save embeddings
    output_path = 'data/embeddings.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"\n✓ Embeddings saved to: {output_path}")
    
    # Print summary
    print("\nEmbedding Summary:")
    for name, emb in embeddings.items():
        print(f"  {name}: {emb.shape[0]} posts × {emb.shape[1]} dimensions")

if __name__ == '__main__':
    main()