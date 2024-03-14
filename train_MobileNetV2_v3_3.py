import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import csv
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)


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


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ChickenDataset('./Detect_chicken_sex/train', transform=transform)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to have the same size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ChickenDataset('./Detect_chicken_sex/test', transform=test_transform)
valid_dataset = ChickenDataset('./Detect_chicken_sex/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


model = mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 2)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train_model(model, criterion, optimizer, train_loader, valid_loader, test_loader, epochs=25, log_file='training_log.csv'):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss','Valid Loss',
                        'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score',
                        'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1-Score',
                        'Valid Accuracy', 'Valid Precision', 'Valid Recall', 'Valid F1-Score',
                        'Confusion Matrix'])

    for epoch in range(epochs):
        model.train()
        train_loss, valid_loss, test_loss = 0.0, 0.0, 0.0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        # Test loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        test_loss /= len(test_loader.dataset)

        train_accuracy = correct_train / total_train
        test_accuracy = correct_test / total_test

        y_true_test = []
        y_pred_test = []
        y_true_train = []
        y_pred_train = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())

        cm_test = confusion_matrix(y_true_test, y_pred_test)
        cm_train = confusion_matrix(y_true_train, y_pred_train)

        report_test = classification_report(y_true_test, y_pred_test, digits=5, output_dict=True)
        report_train = classification_report(y_true_train, y_pred_train, digits=5, output_dict=True)

        test_precision = report_test['macro avg']['precision']
        test_recall = report_test['macro avg']['recall']
        test_f1_score = report_test['macro avg']['f1-score']

        train_precision = report_train['macro avg']['precision']
        train_recall = report_train['macro avg']['recall']
        train_f1_score = report_train['macro avg']['f1-score']

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Test Accuracy: {test_accuracy:.5f}, Test Precision: {test_precision:.5f}, Test Recall: {test_recall:.5f}, Test F1-Score: {test_f1_score:.5f}, Train Accuracy: {train_accuracy:.5f}, Train Precision: {train_precision:.5f}, Train Recall: {train_recall:.5f}, Train F1-Score: {train_f1_score:.5f}, Confusion Matrix Test: {cm_test}, Confusion Matrix Train: {cm_train}')

        cm_test = str(cm_test).replace('\n', ' ')
        cm_train = str(cm_train).replace('\n', ' ')

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, test_loss,
                            valid_loss, test_accuracy, test_precision, test_recall, test_f1_score,
                            train_accuracy, train_precision, train_recall, train_f1_score,
                            cm_test])

# Call to train_model
train_model(model, criterion, optimizer, train_loader, valid_loader, test_loader, epochs=50)


torch.save(model.state_dict(), 'mobilenet_v2_chicken_gender.pth')
