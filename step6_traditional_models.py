"""
Step 6: Train Traditional Topic Models (LDA, LSI, NMF) for All Groups
This prepares traditional models data for final comparison in Step 7

Usage:
    python step6_traditional_models.py                      # Train all models for all groups
    python step6_traditional_models.py -g All               # Train all models for All group only
    python step6_traditional_models.py -m LDA,LSI           # Train only LDA and LSI for all groups
    python step6_traditional_models.py -g Education -m LDA  # Train only LDA for Education group
"""
import pickle
import pandas as pd
import os
import json
import argparse
import sys
from gensim import corpora
from gensim.models import LdaModel, LsiModel
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Import unified metrics utilities
from utils_metrics import calculate_coherence_cv_traditional, calculate_irbo_traditional

def train_lda(texts, n_topics=10):
    """Train LDA model"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=100,
        passes=10,
        per_word_topics=True
    )
    
    # Extract topic words
    topic_words = []
    for topic_id in range(n_topics):
        words = [word for word, _ in lda_model.show_topic(topic_id, topn=10)]
        topic_words.append(words)
    
    return lda_model, topic_words

def train_lsi(texts, n_topics=10):
    """Train LSI model"""
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    lsi_model = LsiModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics
    )
    
    # Extract topic words
    topic_words = []
    for topic_id in range(n_topics):
        words = [word for word, _ in lsi_model.show_topic(topic_id, topn=10)]
        topic_words.append(words)
    
    return lsi_model, topic_words

def train_nmf(texts, n_topics=10, verbose=False):
    """Train NMF model with detailed logging"""
    if verbose:
        print(f"    [NMF] Creating TF-IDF vectorizer (max_features=1000)...")
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    if verbose:
        print(f"    [NMF] Transforming {len(texts)} documents to TF-IDF...")
    tfidf = vectorizer.fit_transform(texts)
    
    if verbose:
        print(f"    [NMF] TF-IDF matrix shape: {tfidf.shape}")
        print(f"    [NMF] Training NMF model with {n_topics} topics...")
    
    nmf_model = SklearnNMF(
        n_components=n_topics,
        random_state=100,
        max_iter=200,
        init='nndsvda',  # Better initialization
        solver='cd',  # Coordinate Descent solver
        verbose=1 if verbose else 0
    )
    
    nmf_model.fit(tfidf)
    
    if verbose:
        print(f"    [NMF] Training completed. Reconstruction error: {nmf_model.reconstruction_err_:.4f}")
    
    # Extract topic words
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        words = [feature_names[i] for i in top_indices]
        topic_words.append(words)
        if verbose and topic_idx < 3:  # Show first 3 topics
            print(f"    [NMF] Topic {topic_idx}: {', '.join(words[:5])}")
    
    return nmf_model, topic_words

def find_best_k(texts, model_type='lda', k_range=range(3, 31), verbose=False):
    """
    Grid search for optimal number of topics
    Returns results in format consistent with BERTopic grid search
    """
    print(f"  Grid search for {model_type.upper()} (k={min(k_range)}-{max(k_range)})...")
    best_k = None
    best_coherence = -1
    results = []
    
    total_k = len(k_range)
    for idx, k in enumerate(k_range, 1):
        if verbose:
            print(f"    [{idx}/{total_k}] Testing k={k}...")
        
        try:
            if model_type == 'lda':
                model, topic_words = train_lda(texts, n_topics=k)
            elif model_type == 'lsi':
                model, topic_words = train_lsi(texts, n_topics=k)
            elif model_type == 'nmf':
                model, topic_words = train_nmf(texts, n_topics=k, verbose=verbose)
            
            if verbose:
                print(f"    [{idx}/{total_k}] Calculating coherence...")
            
            coherence = calculate_coherence_cv_traditional(texts, topic_words)
            irbo = calculate_irbo_traditional(topic_words, top_n=10)
            
            # Format consistent with BERTopic grid search
            results.append({
                'n_topics': k,           # Match BERTopic column name
                'coherence_cv': coherence,  # Match BERTopic column name
                'irbo': irbo,             # Topic diversity score
                'n_posts': len(texts)     # For reference
            })
            
            if verbose:
                print(f"    [{idx}/{total_k}] k={k}, coherence={coherence:.4f}")
            
            if coherence > best_coherence:
                best_coherence = coherence
                best_k = k
        except Exception as e:
            print(f"    [{idx}/{total_k}] Error with k={k}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    print(f"    Best: k={best_k}, coherence={best_coherence:.4f}")
    return best_k, results

# Note: Plotting functions have been moved to step7_final_comparison.py

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train traditional topic models (LDA, LSI, NMF) with grid search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python step6_traditional_models.py                      # Train all models for all groups
    python step6_traditional_models.py -g All               # Train all models for All group only
    python step6_traditional_models.py -m LDA,LSI           # Train only LDA and LSI for all groups
    python step6_traditional_models.py -g Education -m LDA  # Train only LDA for Education group
    python step6_traditional_models.py --models NMF         # Train only NMF for all groups
        """
    )
    parser.add_argument(
        '-g', '--group',
        type=str,
        default=None,
        choices=['All', 'Education', 'Humanities', 'Medicine'],
        help='Specify which group to train (default: train all groups)'
    )
    parser.add_argument(
        '-m', '--models',
        type=str,
        default='LDA,LSI,NMF',
        help='Comma-separated list of models to train (default: LDA,LSI,NMF)'
    )
    return parser.parse_args()

def main():
    """
    Main function to train traditional models
    Supports: LDA, LSI, NMF
    """
    # Parse command line arguments
    args = parse_args()
    
    # Parse model list
    model_list = [m.strip().upper() for m in args.models.split(',')]
    
    # Validate models
    valid_models = {'LDA', 'LSI', 'NMF'}
    invalid_models = set(model_list) - valid_models
    if invalid_models:
        print(f"❌ Error: Invalid models: {invalid_models}")
        print(f"Valid models: {valid_models}")
        sys.exit(1)
    
    print("="*80)
    print("Step 6: Train Traditional Topic Models")
    print("="*80)
    
    if args.group:
        print(f"🎯 Target: {args.group} group")
    else:
        print(f"🎯 Target: All groups")
    
    print(f"📊 Models: {', '.join(model_list)}")
    
    # Load preprocessed data
    print("\n[1/4] Loading preprocessed data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        all_groups = pickle.load(f)
    
    # Filter groups based on command line argument
    if args.group:
        if args.group not in all_groups:
            print(f"\n❌ Error: Group '{args.group}' not found in data!")
            print(f"Available groups: {list(all_groups.keys())}")
            sys.exit(1)
        groups = {args.group: all_groups[args.group]}
        print(f"   Loaded {args.group} group: {len(groups[args.group])} posts")
    else:
        groups = all_groups
        print(f"   Loaded all groups: {', '.join([f'{k}({len(v)})' for k, v in groups.items()])}")
    
    # Create directories if not exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load existing results if training single group/model
    print("\n[2/4] Loading existing results (if any)...")
    if args.group or len(model_list) < 3:
        # Load existing results
        if os.path.exists('results/traditional_best_params.json'):
            with open('results/traditional_best_params.json', 'r') as f:
                best_params = json.load(f)
            print(f"   Loaded existing best_params")
        else:
            best_params = {}
        
        if os.path.exists('models/traditional_models_all_groups.pkl'):
            with open('models/traditional_models_all_groups.pkl', 'rb') as f:
                all_results = pickle.load(f)
            print(f"   Loaded existing results")
        else:
            all_results = {}
    else:
        all_results = {}  # Store all results
        best_params = {}  # Store best k for each group-model combination
    
    # Process groups
    print("\n[3/4] Training traditional models...")
    print(f"   Groups: {list(groups.keys())}")
    print(f"   Models: {model_list}")
    
    for group_name, group_df in groups.items():
        print(f"\n{'='*80}")
        print(f"Processing {group_name} Group ({len(group_df)} posts)")
        print(f"{'='*80}")
        
        texts = group_df['cleaned_text'].tolist()
        
        # Initialize group results (preserve existing if loading)
        if group_name not in all_results:
            all_results[group_name] = {}
        group_results = all_results[group_name]
        
        # Determine k_range based on group size
        # max_k = min(30, len(texts) // 50)  # Reasonable upper limit
        # max_k = max(10, max_k)  # At least 10
        k_range = range(3, 15)
        
        # Train traditional models
        for model_type in model_list:
            model_name = model_type.upper()
            print(f"\n{model_name} Model:")
            
            # Enable verbose logging for NMF
            verbose = True
            
            try:
                # Find best k
                best_k, search_results = find_best_k(
                    texts, 
                    model_type=model_name.lower(), 
                    k_range=k_range, 
                    verbose=verbose
                )
                
                # Train final model with best k
                if model_name == 'LDA':
                    model, topic_words = train_lda(texts, n_topics=best_k)
                elif model_name == 'LSI':
                    model, topic_words = train_lsi(texts, n_topics=best_k)
                elif model_name == 'NMF':
                    model, topic_words = train_nmf(texts, n_topics=best_k)
                
                coherence = calculate_coherence_cv_traditional(texts, topic_words)
                irbo = calculate_irbo_traditional(topic_words, top_n=10)
                
                # Store results (format consistent with BERTopic)
                group_results[model_name] = {
                    'n_topics': best_k,
                    'coherence_cv': coherence,
                    'irbo': irbo,  # Topic diversity score
                    'n_posts': len(texts),
                    'best_k': best_k,  # For backward compatibility
                    'grid_search': search_results  # Store for step7 compatibility
                }
                
                # Store best params
                best_params[f"{group_name}_{model_name}"] = best_k
                
                # Save model
                model_path = f'models/traditional_{model_name.lower()}_{group_name.lower()}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({'model': model, 'topic_words': topic_words}, f)
                
                # Save grid search results (format consistent with BERTopic)
                grid_df = pd.DataFrame(search_results)
                grid_df.to_csv(f'results/grid_search_{model_name.lower()}_{group_name.lower()}.csv', index=False)
                
                print(f"  ✓ {model_name} Results:")
                print(f"      Topics: {best_k}")
                print(f"      Coherence: {coherence:.4f}")
                print(f"      IRBO: {irbo:.4f}")
                print(f"      Saved: {model_path}")
            
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results
    print("\n[4/4] Saving results...")
    
    # Save best parameters
    with open('results/traditional_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("  ✓ Saved: results/traditional_best_params.json")
    
    # Save results summary (format consistent with bertopic_results_summary.csv)
    summary_data = []
    for group_name, models in all_results.items():
        for model_name, metrics in models.items():
            summary_data.append({
                'group': group_name,
                'model': model_name,
                'n_topics': metrics['n_topics'],
                'coherence_cv': metrics['coherence_cv'],
                'irbo': metrics.get('irbo', 1.0),  # Default to 1.0 if not computed
                'n_posts': metrics['n_posts']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    summary_df.to_csv('results/traditional_results_summary.csv', index=False)
    print("  ✓ Saved: results/traditional_results_summary.csv")
    
    # Save detailed results (for step7 comparison)
    with open('models/traditional_models_all_groups.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("  ✓ Saved: models/traditional_models_all_groups.pkl")
    
    # Print summary
    print("\n" + "="*80)
    print("TRADITIONAL MODELS - FINAL RESULTS")
    print("="*80)
    
    if args.group or len(model_list) < 3:
        # Show only trained group/models
        filtered_df = summary_df
        if args.group:
            filtered_df = filtered_df[filtered_df['group'] == args.group]
        if len(model_list) < 3:
            filtered_df = filtered_df[filtered_df['model'].isin(model_list)]
        
        # Format output for clarity
        for group in filtered_df['group'].unique():
            group_data = filtered_df[filtered_df['group'] == group]
            print(f"\n{group} Group:")
            print("-" * 60)
            for _, row in group_data.iterrows():
                irbo_val = row.get('irbo', 1.0)
                irbo_str = f"{irbo_val:.4f}" if pd.notna(irbo_val) else "N/A"
                print(f"  {row['model']:8s} - Topics: {int(row['n_topics']):2d}, Coherence: {row['coherence_cv']:.4f}, IRBO: {irbo_str}")
    else:
        # Show all groups
        for group in summary_df['group'].unique():
            group_data = summary_df[summary_df['group'] == group]
            print(f"\n{group} Group:")
            print("-" * 60)
            for _, row in group_data.iterrows():
                irbo_val = row.get('irbo', 1.0)
                irbo_str = f"{irbo_val:.4f}" if pd.notna(irbo_val) else "N/A"
                print(f"  {row['model']:8s} - Topics: {int(row['n_topics']):2d}, Coherence: {row['coherence_cv']:.4f}, IRBO: {irbo_str}")
    
    # Paper reference comparison for All group
    if 'All' in summary_df['group'].values:
        print("\n" + "="*80)
        print("COMPARISON WITH PAPER (All Group)")
        print("="*80)
        all_data = summary_df[summary_df['group'] == 'All']
        
        paper_results = {
            'LDA': {'topics': 6, 'coherence': 0.542},
            'LSI': {'topics': 8, 'coherence': 0.459},
            'NMF': {'topics': 6, 'coherence': 0.660}
        }
        
        for model in ['LDA', 'LSI', 'NMF']:
            if model in all_data['model'].values:
                row = all_data[all_data['model'] == model].iloc[0]
                paper = paper_results.get(model, {})
                
                if paper:
                    topics_diff = int(row['n_topics']) - paper['topics']
                    coh_diff = row['coherence_cv'] - paper['coherence']
                    coh_pct = (coh_diff / paper['coherence']) * 100
                    
                    print(f"\n{model}:")
                    print(f"  Paper:  Topics={paper['topics']:2d}, Coherence={paper['coherence']:.3f}")
                    print(f"  Ours:   Topics={int(row['n_topics']):2d}, Coherence={row['coherence_cv']:.3f}")
                    print(f"  Diff:   Topics={topics_diff:+3d}, Coherence={coh_diff:+.3f} ({coh_pct:+.1f}%)")
        
        print("="*80)
    
    print(f"\n✓ {'All traditional models' if not args.group else args.group + ' models'} trained successfully!")
    print(f"  - Models: models/traditional_*")
    print(f"  - Results: results/traditional_results_summary.csv")
    print(f"  - Best params: results/traditional_best_params.json")
    
    # Show command to run comparison
    if args.group or len(model_list) < 3:
        print(f"\n💡 Tip: Run full comparison to see detailed results:")
        print(f"    python step7_paper_comparison.py")

if __name__ == '__main__':
    main()