"""
Step 2: Text Preprocessing
"""
import pandas as pd
import pickle
import re
import contractions
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
# print nltk version
print("nltk version:", nltk.__version__)
# paper version: 3.8.1
# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocessing pipeline: expand → clean → lowercase → POS → stopwords
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    doc = nlp(text)
    allowed_pos = {'NOUN', 'ADJ', 'VERB', 'ADV'}
    tokens = [token.text for token in doc if token.pos_ in allowed_pos]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def main():
    print("="*60)
    print("Step 2: Text Preprocessing")
    print("="*60)
    
    # Load groups
    print("\n[1/2] Loading raw groups...")
    with open('data/groups_raw.pkl', 'rb') as f:
        groups = pickle.load(f)
    
    # Preprocess all groups
    print("\n[2/2] Preprocessing text for all groups...")
    for name, group_df in groups.items():
        print(f"\nProcessing {name} group ({len(group_df)} posts)...")
        tqdm.pandas(desc=f"{name}")
        
        group_df = group_df.copy()
        group_df['cleaned_text'] = group_df['post_text'].progress_apply(preprocess_text)
        
        # Remove empty texts
        group_df = group_df[group_df['cleaned_text'].str.len() > 0].copy()
        groups[name] = group_df
        print(f"  After cleaning: {len(group_df)} posts")
    
    # Save preprocessed groups
    output_path = 'data/groups_preprocessed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(groups, f)
    print(f"\n✓ Preprocessed groups saved to: {output_path}")

if __name__ == '__main__':
    main()