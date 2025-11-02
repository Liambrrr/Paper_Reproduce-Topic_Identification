"""
Visualize Grid Search Results
Generate trend plots showing n_neighbors vs topics and coherence
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_grid_search_results():
    """Load all grid search results"""
    results = {}
    groups = ['All', 'Education', 'Humanities', 'Medicine']
    
    for group in groups:
        csv_path = f'results/grid_search_bertopic_{group.lower()}.csv'
        if os.path.exists(csv_path):
            results[group] = pd.read_csv(csv_path)
        else:
            print(f"Warning: {csv_path} not found")
    
    return results

def plot_grid_search_results(results):
    """Create comprehensive visualization of grid search results"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create subplots: 2 rows x 2 columns for each group
    fig = plt.figure(figsize=(18, 14))
    
    groups = ['All', 'Education', 'Humanities', 'Medicine']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: n_neighbors vs n_topics for all groups
    ax1 = plt.subplot(2, 2, 1)
    for i, group in enumerate(groups):
        if group in results:
            df = results[group]
            ax1.plot(df['n_neighbors'], df['n_topics'], 
                    marker='o', linewidth=2, markersize=8,
                    label=group, color=colors[i])
    ax1.set_xlabel('n_neighbors', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Topics', fontsize=12, fontweight='bold')
    ax1.set_title('n_neighbors vs Number of Topics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: n_neighbors vs coherence for all groups
    ax2 = plt.subplot(2, 2, 2)
    for i, group in enumerate(groups):
        if group in results:
            df = results[group]
            ax2.plot(df['n_neighbors'], df['coherence_cv'], 
                    marker='s', linewidth=2, markersize=8,
                    label=group, color=colors[i])
    ax2.set_xlabel('n_neighbors', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coherence Score (C_v)', fontsize=12, fontweight='bold')
    ax2.set_title('n_neighbors vs Coherence Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: n_neighbors vs IRBO for all groups
    ax3 = plt.subplot(2, 2, 3)
    for i, group in enumerate(groups):
        if group in results:
            df = results[group]
            ax3.plot(df['n_neighbors'], df['irbo'], 
                    marker='^', linewidth=2, markersize=8,
                    label=group, color=colors[i])
    ax3.set_xlabel('n_neighbors', fontsize=12, fontweight='bold')
    ax3.set_ylabel('IRBO Score', fontsize=12, fontweight='bold')
    ax3.set_title('n_neighbors vs Topic Diversity (IRBO)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Topics vs Coherence line plot (sorted by n_topics)
    ax4 = plt.subplot(2, 2, 4)
    for i, group in enumerate(groups):
        if group in results:
            df = results[group].copy()
            # Sort by number of topics for smooth line
            df_sorted = df.sort_values('n_topics')
            ax4.plot(df_sorted['n_topics'], df_sorted['coherence_cv'], 
                    marker='D', linewidth=2, markersize=8,
                    label=group, color=colors[i])
    ax4.set_xlabel('Number of Topics', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Coherence Score (C_v)', fontsize=12, fontweight='bold')
    ax4.set_title('Topics vs Coherence (Quality Trade-off)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/grid_search_bertopic_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/grid_search_bertopic_trends.png")
    plt.close()
    
    # Create individual detailed plots for each group
    # Load best parameters to add annotation
    with open('results/best_params_bertopic.json', 'r') as f:
        best_params = json.load(f)
    
    for group in groups:
        if group not in results:
            continue
            
        df = results[group]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get min_cluster_size for this group
        if isinstance(best_params.get(group), dict):
            min_cluster_size = best_params[group]['min_cluster_size']
            fig.suptitle(f'Grid Search Results - {group} Group\n(min_cluster_size={min_cluster_size}, varying n_neighbors)', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Grid Search Results - {group} Group', fontsize=16, fontweight='bold')
        
        # Subplot 1: n_neighbors vs n_topics
        axes[0, 0].plot(df['n_neighbors'], df['n_topics'], 
                       marker='o', linewidth=2.5, markersize=10, color='#1f77b4')
        axes[0, 0].set_xlabel('n_neighbors', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Topics', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('n_neighbors vs Number of Topics', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark best n_neighbors
        best_idx = df['coherence_cv'].idxmax()
        best_n = df.loc[best_idx, 'n_neighbors']
        best_topics = df.loc[best_idx, 'n_topics']
        axes[0, 0].axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_n}')
        axes[0, 0].legend()
        
        # Subplot 2: n_neighbors vs coherence
        axes[0, 1].plot(df['n_neighbors'], df['coherence_cv'], 
                       marker='s', linewidth=2.5, markersize=10, color='#ff7f0e')
        axes[0, 1].set_xlabel('n_neighbors', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Coherence Score (C_v)', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('n_neighbors vs Coherence', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark best coherence
        best_coh = df.loc[best_idx, 'coherence_cv']
        axes[0, 1].axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_n}')
        axes[0, 1].axhline(best_coh, color='green', linestyle='--', alpha=0.5, label=f'Max: {best_coh:.4f}')
        axes[0, 1].legend()
        
        # Subplot 3: n_neighbors vs IRBO
        axes[1, 0].plot(df['n_neighbors'], df['irbo'], 
                       marker='^', linewidth=2.5, markersize=10, color='#2ca02c')
        axes[1, 0].set_xlabel('n_neighbors', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('IRBO Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('n_neighbors vs Topic Diversity', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(best_n, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_n}')
        axes[1, 0].legend()
        
        # Subplot 4: Summary table
        axes[1, 1].axis('off')
        table_data = []
        for _, row in df.iterrows():
            is_best = row['n_neighbors'] == best_n
            marker = '★' if is_best else ''
            table_data.append([
                f"{marker}{int(row['n_neighbors'])}",
                f"{int(row['n_topics'])}",
                f"{row['coherence_cv']:.4f}",
                f"{row['irbo']:.4f}"
            ])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['n_neighbors', 'Topics', 'Coherence', 'IRBO'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best row
        best_row_idx = best_idx + 1  # +1 because of header
        for i in range(4):
            table[(best_row_idx, i)].set_facecolor('#ffeb3b')
        
        plt.tight_layout()
        plt.savefig(f'results/grid_search_bertopic_{group.lower()}_detailed.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/grid_search_bertopic_{group.lower()}_detailed.png")
        plt.close()

def generate_summary_report(results):
    """Generate a summary report of the grid search"""
    
    # Load best parameters
    with open('results/best_params_bertopic.json', 'r') as f:
        best_params = json.load(f)
    
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY REPORT")
    print("="*80)
    
    for group in ['All', 'Education', 'Humanities', 'Medicine']:
        if group not in results:
            continue
        
        df = results[group]
        # Handle both old format (int) and new format (dict)
        if isinstance(best_params.get(group), dict):
            best_n = best_params[group]['n_neighbors']
            best_min_cluster = best_params[group]['min_cluster_size']
        else:
            best_n = best_params.get(group, 15)
            best_min_cluster = 10  # default value
        
        print(f"\n{group} Group:")
        print("-" * 60)
        
        # Find best result
        best_row = df[df['n_neighbors'] == best_n].iloc[0]
        
        print(f"  Best n_neighbors: {best_n}")
        print(f"  Best min_cluster_size: {best_min_cluster}")
        print(f"  Number of Topics: {int(best_row['n_topics'])}")
        print(f"  Coherence Score:  {best_row['coherence_cv']:.4f}")
        print(f"  IRBO Score:       {best_row['irbo']:.4f}")
        print(f"  Outliers:         {int(best_row['n_outliers'])}")
        
        # Show range
        print(f"\n  Range explored:")
        print(f"    n_neighbors:  {df['n_neighbors'].min():.0f} - {df['n_neighbors'].max():.0f}")
        print(f"    Topics:       {df['n_topics'].min():.0f} - {df['n_topics'].max():.0f}")
        print(f"    Coherence:    {df['coherence_cv'].min():.4f} - {df['coherence_cv'].max():.4f}")
        print(f"    IRBO:         {df['irbo'].min():.4f} - {df['irbo'].max():.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON WITH PAPER RESULTS (All group):")
    print("="*80)
    if 'All' in results:
        df = results['All']
        # Handle both old format (int) and new format (dict)
        if isinstance(best_params.get('All'), dict):
            best_n = best_params['All']['n_neighbors']
            best_min_cluster = best_params['All']['min_cluster_size']
            param_str = f"n_neighbors={best_n}, min_cluster_size={best_min_cluster}"
        else:
            best_n = best_params.get('All', 15)
            best_min_cluster = 10
            param_str = f"n_neighbors={best_n}"
        
        best_row = df[df['n_neighbors'] == best_n].iloc[0]
        
        print(f"Paper reported:")
        print(f"  Topics: ~50-57")
        print(f"  Coherence: ~0.616")
        print(f"  IRBO: ~1.0")
        print(f"\nOur result (with {param_str}):")
        print(f"  Topics: {int(best_row['n_topics'])}")
        print(f"  Coherence: {best_row['coherence_cv']:.4f}")
        print(f"  IRBO: {best_row['irbo']:.4f}")
    print("="*80)

def main():
    print("="*80)
    print("Visualizing Grid Search Results")
    print("="*80)
    
    # Check if results exist
    if not os.path.exists('results/best_params_bertopic.json'):
        print("\n❌ Error: Grid search results not found.")
        print("Please run step4_train_bertopic.py first.")
        return
    
    # Load results
    print("\n[1/3] Loading grid search results...")
    results = load_grid_search_results()
    
    if not results:
        print("❌ No grid search results found!")
        return
    
    print(f"Loaded results for {len(results)} groups")
    
    # Create visualizations
    print("\n[2/3] Generating visualizations...")
    plot_grid_search_results(results)
    
    # Generate summary report
    print("\n[3/3] Generating summary report...")
    generate_summary_report(results)
    
    print("\n✓ Visualization complete!")
    print("\nGenerated files:")
    print("  - results/grid_search_bertopic_trends.png (overview)")
    print("  - results/grid_search_bertopic_all_detailed.png")
    print("  - results/grid_search_bertopic_education_detailed.png")
    print("  - results/grid_search_bertopic_humanities_detailed.png")
    print("  - results/grid_search_bertopic_medicine_detailed.png")

if __name__ == '__main__':
    main()