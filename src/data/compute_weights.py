import pandas as pd
import numpy as np
import torch
from pathlib import Path

# The official SOTA ordering for your 17 classes
STAGES = [
    'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 
    't9+', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'Unknown'
]

def compute_sota_weights(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Analyzing CSV with columns: {df.columns.tolist()}")

    # 1. Map 'Event' strings to their index in the STAGES list
    if 'Event' not in df.columns:
        raise KeyError(f"Could not find 'Event' column. Found: {df.columns.tolist()}")
    
    # Create a mapping dictionary: {'tPB2': 0, 'tPNa': 1, ...}
    stage_to_idx = {stage: i for i, stage in enumerate(STAGES)}
    
    # Map the Event column to a new numeric column
    df['stage_idx'] = df['Event'].map(stage_to_idx)
    
    # Check for any unmapped labels (typos in CSV)
    if df['stage_idx'].isnull().any():
        missing = df[df['stage_idx'].isnull()]['Event'].unique()
        print(f"[WARNING] Unrecognized labels in CSV: {missing}")
        df['stage_idx'] = df['stage_idx'].fillna(len(STAGES)-1) # Default to 'Unknown'

    # 2. Calculate counts using the numeric indices
    counts = np.bincount(df['stage_idx'].astype(int), minlength=len(STAGES))
    
    # Handle zero counts (for rare classes like tHB) to avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    
    # 3. Inverse Frequency Weighting
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    
    # Normalize so the average weight is 1.0
    weights = weights / weights.mean()
    
    return weights

if __name__ == "__main__":
    csv_file = Path("data/splits/train.csv")
    
    if csv_file.exists():
        final_weights = compute_sota_weights(csv_file)
        print("\n" + "="*40)
        print("      SOTA CLASS WEIGHTS GENERATED")
        print("="*40)
        # Rounding for clean copy-pasting
        formatted = [round(w, 4) for w in final_weights.tolist()]
        print(f"CLASS_WEIGHTS = {formatted}")
        print("="*40)
        print("\nACTION: Copy the list above into your trainer.py configuration.")
    else:
        print(f"Error: Could not find {csv_file}")