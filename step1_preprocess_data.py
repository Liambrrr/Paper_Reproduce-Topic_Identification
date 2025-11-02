"""
Step 1: Data Loading, Cleaning and Grouping
"""
import pandas as pd
import pickle
import os

def main():
    print("="*60)
    print("Step 1: Data Loading and Preprocessing")
    print("="*60)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    data_path = 'dataset/stanfordMOOCForumPostsSet.xlsx'
    df = pd.read_excel(data_path)
    print(f"Total posts: {len(df)}")
    
    # Rename columns
    df['urgency'] = df['Urgency(1-7)']
    df['course_type'] = df['CourseType']
    df['post_text'] = df['Text']
    
    # Remove missing text
    print(f"\n[2/4] Cleaning data...")
    df = df[df['post_text'].notna()].copy()
    print(f"After removing missing text: {len(df)} posts")
    
    # Filter urgent posts (urgency > 4)
    print(f"\n[3/4] Filtering urgent posts (urgency >= 4)...")
    df_urgent = df[df['urgency'] >= 4].copy()
    print(f"Urgent posts: {len(df_urgent)}")
    
    # Create analysis groups
    print(f"\n[4/4] Creating analysis groups...")
    groups = {
        'All': df_urgent,
        'Education': df_urgent[df_urgent['course_type'] == 'Education'],
        'Humanities': df_urgent[df_urgent['course_type'] == 'Humanities'],
        'Medicine': df_urgent[df_urgent['course_type'] == 'Medicine']
    }
    
    for name, group_df in groups.items():
        print(f"  {name}: {len(group_df)} posts")
    
    # Save groups
    os.makedirs('data', exist_ok=True)
    output_path = 'data/groups_raw.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(groups, f)
    print(f"\n✓ Groups saved to: {output_path}")

if __name__ == '__main__':
    main()