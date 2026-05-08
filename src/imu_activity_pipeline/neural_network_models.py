"""Neural network building blocks for IMU window classification.

Purpose:
    Defines the loss functions and PyTorch models used by training and inference,
    including focal loss, triplet loss, binary detection, activity classification,
    and the combined background-plus-activity classifier.
Inputs:
    Receives normalized IMU windows shaped as `(batch, window_size, channels)`
    and optional embedding/class labels during training.
Outputs:
    Produces class logits and, when requested, embeddings for metric-learning
    losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== Loss Functions ============================================

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced window classification."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # per-class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class TripletLoss(nn.Module):
    """Triplet loss for separating class embeddings.

    The loss pulls same-class embeddings together and pushes different-class
    embeddings apart, which helps distinguish visually similar motion patterns.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (batch, dim)
        labels: (batch,)
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=device)

        # Pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(batch_size):
            # Positive: same class, different sample
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=device) != i)
            # Negative: different class
            neg_mask = labels != labels[i]

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_dists = dist_matrix[i][pos_mask]
            neg_dists = dist_matrix[i][neg_mask]

            # Hard mining: hardest positive, hardest negative
            hardest_pos = pos_dists.max()
            hardest_neg = neg_dists.min()

            triplet = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += triplet
            count += 1

        return loss / max(count, 1)


# ==================== Stage 1: Binary Detector ==================================

class Stage1Detector(nn.Module):
    """Binary classifier: Activity vs Background.
    - 1D-CNN branch (local patterns)
    - Hand-crafted feature branch (statistics)
    - Feature fusion -> Bi-LSTM -> Sigmoid
    """

    def __init__(self, input_channels=6, handcraft_dim=0, window_size=300):
        super().__init__()
        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )

        # Hand-crafted feature branch
        self.handcraft_dim = handcraft_dim
        if handcraft_dim > 0:
            self.handcraft_fc = nn.Sequential(
                nn.Linear(handcraft_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            fusion_dim = 128 + 64
        else:
            fusion_dim = 128

        # Fusion
        self.fusion_fc = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(0.3)

        # Bi-LSTM
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.3)

        # Output
        self.fc_out = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x, handcraft_feats=None):
        """
        x: (batch, window_size, channels)
        handcraft_feats: (batch, handcraft_dim) or None
        """
        # CNN
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn(x_cnn)  # (batch, 128, L/4)

        # Global average pool for fusion
        cnn_global = x_cnn.mean(dim=2)  # (batch, 128)

        # Fuse with handcraft features
        if self.handcraft_dim > 0 and handcraft_feats is not None:
            hf = self.handcraft_fc(handcraft_feats)  # (batch, 64)
            fused = torch.cat([cnn_global, hf], dim=1)
        else:
            fused = cnn_global

        fused = F.relu(self.fusion_fc(fused))  # (batch, 256)
        fused = self.dropout(fused)

        # LSTM on CNN feature map
        x_seq = x_cnn.permute(0, 2, 1)  # (batch, L/4, 128)
        lstm_out, _ = self.lstm(x_seq)  # (batch, L/4, 256)
        lstm_feat = lstm_out[:, -1, :]  # (batch, 256)

        # Combine
        combined = fused + lstm_feat  # residual
        return self.fc_out(combined).squeeze(-1)


# ==================== Stage 2: Activity Classifier ==============================

class Stage2Classifier(nn.Module):
    """5-class activity classifier.
    - Multi-scale CNN branches (k=3, k=7, k=15)
    - Bi-LSTM temporal encoding
    - Attention pooling for segment-level features
    - Classification head with embedding output for Triplet Loss
    """

    def __init__(self, input_channels=6, num_classes=5, window_size=300):
        super().__init__()

        # Multi-scale 1D CNN branches.
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )

        # Merge and further convolution
        self.conv_merge = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Bi-LSTM temporal encoding
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.3)

        # Attention pooling over temporal features.
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Embedding layer for Triplet Loss
        self.embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_embedding=False):
        """
        x: (batch, window_size, channels)
        return_embedding: if True, also return embedding for Triplet Loss
        """
        x = x.permute(0, 2, 1)  # (batch, channels, window_size)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = torch.cat([b1, b2, b3], dim=1)  # (batch, 192, L/2)

        x = self.conv_merge(x)  # (batch, 256, L/8)
        x = x.permute(0, 2, 1)  # (batch, L/8, 256)

        x, _ = self.lstm(x)  # (batch, L/8, 256)

        # Attention pooling
        att_weights = self.attention(x)  # (batch, L/8, 1)
        att_weights = F.softmax(att_weights, dim=1)
        x = torch.sum(x * att_weights, dim=1)  # (batch, 256)

        # Embedding
        emb = self.embedding(x)  # (batch, 128)

        # Classification
        logits = self.classifier(emb)  # (batch, num_classes)

        if return_embedding:
            return logits, emb
        return logits


# ==================== Combined Model (for practical single-model approach) ===========

class CombinedModel(nn.Module):
    """Combined 6-class model (background + 5 activities).
    Uses multi-scale CNN + BiLSTM architecture.
    Also outputs embeddings for Triplet Loss.
    """

    def __init__(self, input_channels=6, num_classes=6, window_size=300):
        super().__init__()

        # Multi-scale 1D CNN
        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )

        self.conv_merge = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.lstm = nn.LSTM(192, 128, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.3)

        # Combined classifier (embedding + classification head)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_embedding=False):
        xp = x.permute(0, 2, 1)  # (batch, channels, window_size)

        b1 = self.branch1(xp)
        b2 = self.branch2(xp)
        b3 = self.branch3(xp)
        merged = torch.cat([b1, b2, b3], dim=1)

        cnn_out = self.conv_merge(merged)
        cnn_feat = cnn_out.squeeze(-1)

        lstm_in = merged.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_feat = lstm_out[:, -1, :]

        combined = torch.cat([cnn_feat, lstm_feat], dim=1)

        if return_embedding:
            # Extract embedding from intermediate layer for triplet loss
            emb = combined
            for layer in list(self.classifier.children())[:5]:
                emb = layer(emb)
            logits = self.classifier(combined)
            return logits, emb

        logits = self.classifier(combined)
        return logits
