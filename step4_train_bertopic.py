"""
Step 4: Train BERTopic Models with Grid Search for n_neighbors and min_cluster_size
This script performs grid search over different n_neighbors and min_cluster_size values to find optimal parameters

Usage:
    python step4_train_bertopic.py              # Train all groups
    python step4_train_bertopic.py --group All  # Train only All group
    python step4_train_bertopic.py -g Education # Train only Education group
"""
import pickle
import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
import json
import argparse
import sys

# Import unified metrics utilities
from utils_metrics import calculate_coherence_cv_bertopic, calculate_irbo

def create_umap_model(n_samples, n_neighbors):
    """Create UMAP model with specified n_neighbors"""
    # Ensure n_neighbors is valid
    n_neighbors = min(n_neighbors, n_samples - 1)
    n_neighbors = max(2, n_neighbors)  # At least 2
    
    return UMAP(
        n_components=5,
        metric='cosine',
        min_dist=0.05,
        random_state=100,
        n_neighbors=n_neighbors
    )

def create_bertopic_model(n_samples, embedding_model, n_neighbors, min_cluster_size):
    """Create BERTopic model with specified n_neighbors and min_cluster_size"""
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    umap_model = create_umap_model(n_samples, n_neighbors)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', prediction_data=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        nr_topics='auto',
        verbose=False  # Set to False for grid search
    )
    
    return topic_model

def grid_search_parameters(name, texts, embs, embedding_model, n_neighbors_range, min_cluster_size_range):
    """Perform grid search over n_neighbors and min_cluster_size values"""
    print(f"\n{'='*60}")
    print(f"Grid Search for {name} group ({len(texts)} posts)")
    print(f"{'='*60}")
    print(f"Testing n_neighbors: {n_neighbors_range}")
    print(f"Testing min_cluster_size: {min_cluster_size_range}")
    print(f"Total combinations: {len(n_neighbors_range) * len(min_cluster_size_range)}")
    
    results = []
    total_combinations = len(n_neighbors_range) * len(min_cluster_size_range)
    current_combination = 0
    
    for min_cluster_size in min_cluster_size_range:
        for n_neighbors in n_neighbors_range:
            current_combination += 1
            print(f"\n  [{current_combination}/{total_combinations}] Testing n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}...")
            
            try:
                # Create and train model
                topic_model = create_bertopic_model(len(texts), embedding_model, n_neighbors, min_cluster_size)
                topics, probs = topic_model.fit_transform(texts, embs)
                
                # Calculate metrics
                n_topics = len(set(topics)) - (1 if -1 in topics else 0)
                n_outliers = sum(1 for t in topics if t == -1)
                
                # Only calculate coherence and IRBO if we have topics
                if n_topics > 0:
                    coherence = calculate_coherence_cv_bertopic(texts, topics, topic_model, top_n=10)
                    irbo = calculate_irbo(topic_model, top_n=10)
                else:
                    coherence = 0.0
                    irbo = 0.0
                
                result = {
                    'n_neighbors': n_neighbors,
                    'min_cluster_size': min_cluster_size,
                    'n_topics': n_topics,
                    'n_outliers': n_outliers,
                    'coherence_cv': coherence,
                    'irbo': irbo
                }
                results.append(result)
                
                print(f"      Topics: {n_topics}, Outliers: {n_outliers}, "
                      f"Coherence: {coherence:.4f}, IRBO: {irbo:.4f}")
            
            except Exception as e:
                print(f"      Error: {str(e)}")
                continue
    
    return results

def select_best_params(results):
    """Select best parameters based on coherence score"""
    if not results:
        print("No results found")
        exit(0)
    
    # Filter out results with 0 or very few topics
    valid_results = [r for r in results if r['n_topics'] >= 2]
    
    if not valid_results:
        # If no valid results, use the one with most topics
        print("No valid results found")
        exit(0)
    
    # Select based on highest coherence score
    best = max(valid_results, key=lambda x: x['coherence_cv'])
    return {
        'n_neighbors': best['n_neighbors'],
        'min_cluster_size': best['min_cluster_size']
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train BERTopic models with grid search over n_neighbors and min_cluster_size',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python step4_train_bertopic.py              # Train all groups
    python step4_train_bertopic.py --group All  # Train only All group
    python step4_train_bertopic.py -g Education # Train only Education group
    python step4_train_bertopic.py -g Humanities # Train only Humanities group
    python step4_train_bertopic.py -g Medicine  # Train only Medicine group
        """
    )
    parser.add_argument(
        '-g', '--group',
        type=str,
        default=None,
        choices=['All', 'Education', 'Humanities', 'Medicine'],
        help='Specify which group to train (default: train all groups)'
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("="*80)
    print("Step 4: Train BERTopic Models with Grid Search")
    print("="*80)
    
    if args.group:
        print(f"\n🎯 Target: {args.group} group only")
    else:
        print(f"\n🎯 Target: All groups")
    
    # Configuration
    # Grid search range for n_neighbors
    # Paper mentions 0.7 but this should be integer values
    # We'll test a range of values from 5 to 30
    N_NEIGHBORS_RANGE = [5, 7, 10, 12, 15, 18, 20, 25, 30]
    
    # Grid search range for min_cluster_size
    # Testing different values to find optimal clustering
    MIN_CLUSTER_SIZE_RANGE = [2,3,4, 5, 10, 15, 20]
    
    # Load preprocessed groups and embeddings
    print("\n[1/4] Loading data...")
    with open('data/groups_preprocessed.pkl', 'rb') as f:
        all_groups = pickle.load(f)
    with open('data/embeddings.pkl', 'rb') as f:
        all_embeddings = pickle.load(f)
    
    # Filter groups based on command line argument
    if args.group:
        if args.group not in all_groups:
            print(f"\n❌ Error: Group '{args.group}' not found in data!")
            print(f"Available groups: {list(all_groups.keys())}")
            sys.exit(1)
        groups = {args.group: all_groups[args.group]}
        embeddings = {args.group: all_embeddings[args.group]}
        print(f"   Loaded {args.group} group: {len(groups[args.group])} posts")
    else:
        groups = all_groups
        embeddings = all_embeddings
        print(f"   Loaded all groups: {', '.join([f'{k}({len(v)})' for k, v in groups.items()])}")
    
    # Load embedding model
    print("\n[2/4] Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Grid search for each group
    print("\n[3/4] Performing grid search...")
    
    # Load existing results if training single group
    if args.group:
        # Load existing best_params if available
        best_params_path = 'results/best_params_bertopic.json'
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            print(f"   Loaded existing best_params, will update {args.group}")
        else:
            best_params = {}
        
        # Load existing grid results if available
        grid_results_path = 'results/grid_search_bertopic_all.json'
        if os.path.exists(grid_results_path):
            with open(grid_results_path, 'r') as f:
                all_grid_results = json.load(f)
            print(f"   Loaded existing grid results, will update {args.group}")
        else:
            all_grid_results = {}
    else:
        all_grid_results = {}
        best_params = {}
    
    for name, group_df in groups.items():
        texts = group_df['cleaned_text'].tolist()
        embs = embeddings[name]
        
        # Perform grid search
        grid_results = grid_search_parameters(
            name, texts, embs, embedding_model, N_NEIGHBORS_RANGE, MIN_CLUSTER_SIZE_RANGE
        )
        all_grid_results[name] = grid_results
        
        # Select best parameters
        best_params_group = select_best_params(grid_results)
        best_params[name] = best_params_group
        
        print(f"\n  ✓ Best parameters for {name}:")
        print(f"      n_neighbors={best_params_group['n_neighbors']}, min_cluster_size={best_params_group['min_cluster_size']}")
        
        # Save full grid search results (all combinations)
        df_full = pd.DataFrame(grid_results)
        df_full.to_csv(f'results/grid_search_bertopic_{name.lower()}_full.csv', index=False)
        print(f"    Full grid search results saved to: results/grid_search_bertopic_{name.lower()}_full.csv")
        
        # Save filtered results for visualization (only best min_cluster_size)
        # This maintains backward compatibility with existing visualization code
        best_min_cluster = best_params_group['min_cluster_size']
        df_filtered = df_full[df_full['min_cluster_size'] == best_min_cluster].copy()
        # Remove min_cluster_size column for backward compatibility
        df_viz = df_filtered[['n_neighbors', 'n_topics', 'n_outliers', 'coherence_cv', 'irbo']].reset_index(drop=True)
        df_viz.to_csv(f'results/grid_search_bertopic_{name.lower()}.csv', index=False)
        print(f"    Filtered results (min_cluster_size={best_min_cluster}) saved to: results/grid_search_bertopic_{name.lower()}.csv")
    
    # Save all grid search results
    with open('results/grid_search_bertopic_all.json', 'w') as f:
        json.dump(all_grid_results, f, indent=2)
    
    with open('results/best_params_bertopic.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\n" + "="*80)
    print("BEST PARAMETERS SELECTED:")
    print("="*80)
    for name, params in best_params.items():
        print(f"  {name}: n_neighbors={params['n_neighbors']}, min_cluster_size={params['min_cluster_size']}")
    print("="*80)
    
    # Train final models with best parameters
    print("\n[4/4] Training final models with best parameters...")
    
    # Load existing results if training single group
    if args.group:
        # Load existing summary results
        summary_path = 'results/bertopic_results_summary.csv'
        if os.path.exists(summary_path):
            existing_results = pd.read_csv(summary_path, index_col=0)
            results = existing_results.to_dict('index')
            print(f"   Loaded existing results, will update {args.group}")
        else:
            results = {}
        
        # Load existing topics and probs
        if os.path.exists('models/topics_dict.pkl'):
            with open('models/topics_dict.pkl', 'rb') as f:
                topics_dict = pickle.load(f)
        else:
            topics_dict = {}
        
        if os.path.exists('models/probs_dict.pkl'):
            with open('models/probs_dict.pkl', 'rb') as f:
                probs_dict = pickle.load(f)
        else:
            probs_dict = {}
    else:
        results = {}
        topics_dict = {}
        probs_dict = {}
    
    topic_models = {}
    
    for name, group_df in groups.items():
        print(f"\n  Training final {name} model...")
        
        texts = group_df['cleaned_text'].tolist()
        embs = embeddings[name]
        best_params_group = best_params[name]
        best_n_neighbors = best_params_group['n_neighbors']
        best_min_cluster_size = best_params_group['min_cluster_size']
        
        # Train model with best parameters
        topic_model = create_bertopic_model(len(texts), embedding_model, best_n_neighbors, best_min_cluster_size)
        topic_model.verbose = True  # Enable verbose for final training
        topics, probs = topic_model.fit_transform(texts, embs)
        
        # Store results
        topic_models[name] = topic_model
        topics_dict[name] = topics
        probs_dict[name] = probs
        
        # Calculate metrics
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        coherence = calculate_coherence_cv_bertopic(texts, topics, topic_model, top_n=10)
        irbo = calculate_irbo(topic_model, top_n=10)
        
        # Store metrics
        results[name] = {
            'n_topics': n_topics,
            'n_outliers': n_outliers,
            'coherence_cv': coherence,
            'irbo': irbo,
            'n_posts': len(texts),
            'n_neighbors': best_n_neighbors,
            'min_cluster_size': best_min_cluster_size
        }
        
        print(f"    Topics: {n_topics}, Outliers: {n_outliers}")
        print(f"    Coherence: {coherence:.4f}, IRBO: {irbo:.4f}")
        
        # Save model
        topic_model.save(f'models/bertopic_{name.lower()}')
        print(f"    Saved: models/bertopic_{name.lower()}")
    
    # Save topics and probs
    with open('models/topics_dict.pkl', 'wb') as f:
        pickle.dump(topics_dict, f)
    with open('models/probs_dict.pkl', 'wb') as f:
        pickle.dump(probs_dict, f)
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df.to_csv('results/bertopic_results_summary.csv')
    
    print(f"\n" + "="*80)
    print("BERTOPIC FINAL RESULTS")
    print("="*80)
    
    if args.group:
        # Show only the trained group
        print(f"\n{args.group} Group Results:")
        print("-" * 60)
        if args.group in results_df.index:
            row = results_df.loc[args.group]
            print(f"  Topics:           {int(row['n_topics'])}")
            print(f"  Outliers:         {int(row['n_outliers'])}")
            print(f"  Coherence:        {row['coherence_cv']:.4f}")
            print(f"  IRBO:             {row['irbo']:.4f}")
            print(f"  n_neighbors:      {int(row['n_neighbors'])}")
            print(f"  min_cluster_size: {int(row['min_cluster_size'])}")
    else:
        # Show all groups
        print(results_df.to_string())
    
    print("\n" + "="*80)
    print("Paper Reference (All group): Topics ~50-57, Coherence ~0.616")
    if 'All' in results_df.index:
        print(f"Our Results (All): Topics {int(results_df.loc['All', 'n_topics'])}, "
              f"Coherence {results_df.loc['All', 'coherence_cv']:.4f}")
    print("="*80)
    
    if args.group:
        print(f"\n✓ {args.group} model trained and saved with optimized parameters")
    else:
        print(f"\n✓ All models trained and saved with optimized parameters")
    print(f"  - Models: models/bertopic_*")
    print(f"  - Results: results/bertopic_results_summary.csv")
    
    # Show command to visualize
    if args.group:
        print(f"\n💡 Tip: Run visualization to see updated results:")
        print(f"    python step4_visualize_grid_search.py")

if __name__ == '__main__':
    main()