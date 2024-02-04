import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import csv

class ChickenDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        image_dir = os.path.join(root_dir, 'images')
        label_dir = os.path.join(root_dir, 'labels')

        for label_name in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_name)
            with open(label_path, 'r') as file:
                label = [float(x) for x in file.read().strip().split()]
            if label:
                self.labels.append(label[0])
                image_path = os.path.join(image_dir, label_name.replace('.txt', '.jpg'))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]  # Đã được sửa đổi ở trên
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def _load_label(self, label_path):
        with open(label_path, 'r') as file:
            label = [float(x) for x in file.read().strip().split()]
        return torch.tensor(label, dtype=torch.float32)
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChickenDataset('./Detect_chicken_sex/train', transform=transform)
test_dataset = ChickenDataset('./Detect_chicken_sex/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=25,log_file='training_log.csv'):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
    for epoch in range(epochs):
        model.train()
        train_loss, test_loss = 0.0, 0.0
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
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)
        accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, test Loss: {test_loss:.5f}, Accuracy: {accuracy:.5f}')
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=5, output_dict=True)
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, test_loss, accuracy, precision, recall, f1_score])
        

# Call to train_model
train_model(model, criterion, optimizer, train_loader, test_loader, epochs=10)

torch.save(model.state_dict(), 'resnet18_chicken_gender.pth')