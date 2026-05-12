import torch
from torchvision import datasets, transforms
from models import AttentionCNNLSTM
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load FER2013 dataset correctly
train_data = datasets.FER2013(
    root='data',
    split='train',
    download=True,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

# Create model
model = AttentionCNNLSTM()

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

print("🚀 Training Started...")

# Training loop
for epoch in range(5):

    running_loss = 0.0

    for images, labels in train_loader:

        # Create fake sequence of 15 frames
        images_seq = images.unsqueeze(1).repeat(1, 15, 1, 1, 1)

        # Forward pass
        outputs = model(images_seq)

        # Convert FER labels into 2 classes
        # 0 = attentive, 1 = distracted
        labels = (labels > 3).long()

        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"✅ Epoch {epoch + 1} completed | Loss: {running_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), 'best_model.pth')

print("🎉 Model trained and saved successfully!")
print("📁 File created: best_model.pth")