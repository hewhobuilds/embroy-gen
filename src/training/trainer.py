import sys
import os
from pathlib import Path


current_file = Path(__file__).resolve()
src_path = current_file.parents[1] 
root_path = current_file.parents[2] 

sys.path.append(str(root_path))
sys.path.append(str(src_path))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset import EmbryoSequenceDataset
from models.hybrid_model import EmbryoGenModel
from training.utils import seed_everything

# --- CONFIGURATION ---
EPOCHS = 30 
BATCH_SIZE = 8 
MAX_LR = 1e-4  
WEIGHT_DECAY = 0.05  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = Path("experiments/run_002_sota_weighted")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- SOTA BALANCED WEIGHTS ---
# We use a log-scale dampening. This prevents weights from being 16.0+
# which causes the 0.0000 accuracy by drowning out the gradient.
CLASS_WEIGHTS = torch.tensor([
    1.5, 0.8, 1.8, 1.0, 2.0, 1.0, 1.8, 1.8, 
    1.8, 1.0, 0.8, 1.2, 1.2, 1.5, 1.2, 4.0, 1.0
]).to(DEVICE)

def train():
    seed_everything(42)
    
    # SOTA AUGMENTATION
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = EmbryoSequenceDataset("data/splits/train.csv", "data/processed/stacked_frames", transform=transform)
    val_ds = EmbryoSequenceDataset("data/splits/val.csv", "data/processed/stacked_frames", transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = EmbryoGenModel(num_classes=17).to(DEVICE)
    
    # Weighted Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler('cuda') 

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for frames, times, labels in loop:
            frames, times, labels = frames.to(DEVICE), times.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(frames, times) 
                loss = criterion(logits.view(-1, 17), labels.view(-1))
            
            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
                
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # Validation with Debugging
        val_acc = validate(model, val_loader, epoch)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_DIR / "best_model_sota.pth")
            print(f">>> Saved SOTA Best Model (Acc: {val_acc:.4f})")

def validate(model, loader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (frames, times, labels) in enumerate(loader):
            frames, times, labels = frames.to(DEVICE), times.to(DEVICE), labels.to(DEVICE)
            logits = model(frames, times)
            preds = torch.argmax(logits, dim=-1)
            
            # --- DEBUG: See what it is predicting if acc is 0 ---
            if epoch == 0 and i == 0:
                print(f"\nDEBUG: Sample Preds: {preds[0][:5].tolist()} | Sample Labels: {labels[0][:5].tolist()}")
                
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    train()