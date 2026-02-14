import torch
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
root_path = current_file.parents[2]
sys.path.append(str(root_path))

from src.data.dataset import EmbryoSequenceDataset
from src.models.hybrid_model import EmbryoGenModel

def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = root_path / "experiments/run_001_hybrid_sota/best_model.pth"
    SAVE_DIR = root_path / "experiments/run_001_hybrid_sota"
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = EmbryoSequenceDataset("data/splits/test.csv", "data/processed/stacked_frames", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # 2. Load Model
    # Note: Ensure the architecture (d_model, nhead, etc.) matches your trainer!
    model = EmbryoGenModel(num_classes=17).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Starting Evaluation on {len(test_ds)} embryos...")
    
    with torch.no_grad():
        for frames, times, labels in test_loader:
            frames, times, labels = frames.to(DEVICE), times.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            logits = model(frames, times) # Shape: (1, 16, 17)
            preds = torch.argmax(logits, dim=-1) # Shape: (1, 16)
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    # 3. Generate Metrics
    # Mapping indices back to class names
    class_names = test_ds.classes
    
    print("\n" + "="*30)
    print("      CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=range(len(class_names))))

    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Embryo Staging: Confusion Matrix')
    plt.xlabel('Predicted Stage')
    plt.ylabel('Actual Stage')
    
    cm_path = SAVE_DIR / "test_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix saved to: {cm_path}")

if __name__ == "__main__":
    evaluate()