import torch
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Custom Dataset class
class ChickenDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
        image = Image.open(img_name)
        
        label = 0  # Giá trị mặc định nếu nhãn không đọc được
        try:
            with open(label_name, 'r') as f:
                label_line = f.readline().strip()
                label_parts = label_line.split()
                if label_parts:  # Kiểm tra xem có dữ liệu trong label_parts không
                    label = int(label_parts[0])
        except FileNotFoundError:
            print(f"Không tìm thấy file nhãn: {label_name}")
        except ValueError:
            print(f"Định dạng nhãn không đúng trong file: {label_name}")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ChickenDataset(image_dir='./Detect chicken sex.v4i.yolov8/train/images', label_dir='./Detect chicken sex.v4i.yolov8/train/labels', transform=transform)
valid_dataset = ChickenDataset(image_dir='./Detect chicken sex.v4i.yolov8/valid/images', label_dir='./Detect chicken sex.v4i.yolov8/valid/labels', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Load the VGG16 model
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Modify the classifier
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=25):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

# Start training
train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10)

# Save the model
torch.save(model.state_dict(), 'vgg16_chicken_gender.pth')
