import torch
import torch.nn as nn
from torchvision import models

class EmbryoGenModel(nn.Module):
    def __init__(self, num_classes=17, d_model=256, nhead=8, num_layers=3):
        super(EmbryoGenModel, self).__init__()
        
        # 1. Feature Extractor (ResNet18)
        # Pre-trained on ImageNet to recognize basic shapes/textures
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # Remove last FC layer
        
        # 2. Linear projection and Stability Layers
        self.feature_proj = nn.Linear(512, d_model)
        
        # --- STABILITY UPGRADE: LayerNorm ---
        # LayerNorm is crucial for Transformers. It keeps the values centered 
        # around 0 and prevents the numbers from getting large enough to cause NaNs.
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        
        # 3. Time Embedder (Normalized)
        self.time_proj = nn.Linear(1, d_model)
        
        # 4. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, d_model))
        
        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024, # Increased for better temporal memory
            dropout=0.2,
            batch_first=True,
            activation='gelu' # GELU is more stable for deep sequence models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 6. Classification Head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, frames, times):
        # frames shape: (Batch, 16, 3, 224, 224)
        # times shape: (Batch, 16)
        
        b, t, c, h, w = frames.shape
        
        # Flatten Batch and Time for ResNet
        x = frames.view(b * t, c, h, w)
        features = self.feature_extractor(x) # (B*T, 512, 1, 1)
        features = features.view(b, t, -1)    # (B, t, 512)
        
        # Project and Normalize
        # We normalize HERE so the Transformer doesn't explode
        x = self.feature_proj(features)
        x = self.norm(x) 
        x = self.dropout(x)
        
        # Add Time Information (TIME NORMALIZATION)
        # Raw hours (e.g., 80.5) are too large for neural weights. 
        # Dividing by 150.0 scales them to roughly 0-1.
        normalized_time = times.unsqueeze(-1) / 150.0 
        time_emb = self.time_proj(normalized_time)
        x = x + time_emb 
        
        # Add Positional Encoding
        x = x + self.pos_embedding
        
        # Transformer Processing
        x = self.transformer(x)               
        
        # Classify each frame
        logits = self.classifier(x)           
        
        return logits