import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class ChickenDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))
                    label_file = file.replace('.jpg', '.txt')
                    self.labels.append(os.path.join(label_dir, label_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        label = self._load_label(label_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def _load_label(self, label_path):
        # Implement your logic to load the label here
        # Return the label as per your requirement
        pass

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChickenDataset('./classfier/train/images', './classfier/train/labels', transform=transform)
test_dataset = ChickenDataset('./classfier/test/images', './classfier/test/labels', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=25):
    for epoch in range(epochs):
        model.train()
        train_loss, valid_loss = 0.0, 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}')

# Call to train_model
train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10)

torch.save(model.state_dict(), 'resnet18_chicken_gender.pth')
