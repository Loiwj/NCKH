import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
class ChickenDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.images = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = img_path.split('/')[-2]  # Tên thư mục là nhãn
        label = 1 if label == 'cock' else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChickenDataset('./classfier/train', transform=transform)
test_dataset = ChickenDataset('./classfier/test', transform=transform)

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
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}, Accuracy: {accuracy:.5f}')
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=5)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
# Call to train_model
train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10)


torch.save(model.state_dict(), 'resnet18_chicken_gender.pth')
