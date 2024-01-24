import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

# Replace with the actual path to your images and labels
train_images_path = 'Determining sex in animal husbandry/Detect chicken sex.v4i.yolov8/train/images'
train_labels_path = 'Determining sex in animal husbandry/Detect chicken sex.v4i.yolov8/train/labels'

# Define transformations for the input data
transformations = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.ImageFolder(
    root=train_images_path,
    transform=transformations
)

# Create a DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

# Load the VGG16 model pre-trained on ImageNet
vgg16 = models.vgg16(pretrained=True)

# Replace the classifier - Example for 2 classes
vgg16.classifier[6] = nn.Linear(4096,2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}')

# Train the model
train_model(vgg16, criterion, optimizer, num_epochs=25)

# Save the model
torch.save(vgg16.state_dict(), 'vgg16_chicken_gender.pth')
