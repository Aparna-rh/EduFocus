import torch
import torch.nn as nn
from torchvision import models

class AttentionCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights='DEFAULT')
        self.cnn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.cnn.last_channel, 512))
        
        self.lstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # attentive / distracted
        )
    
    def forward(self, x):
        # x: batch x seq_len x 3 x 224 x 224
        batch, seq, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq, -1)
        
        lstm_out, _ = self.lstm(features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])  # last timestep
        return out

# Load model function
def load_model(path='best_model.pth'):
    model = AttentionCNNLSTM()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model