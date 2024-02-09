import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import csv
from sklearn.metrics import (
    precision_score,
    recall_score,
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
        label = self.labels[idx]
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
test_dataset = ChickenDataset('./Detect_chicken_sex/test', transform=transform)
valid_dataset = ChickenDataset('./Detect_chicken_sex/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
valid_loader = DataLoader(valid_dataset, batch_size=32)

model = resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 2)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class_names = ['cook', 'hen']
# Hàm train_model
def train_model(model, criterion, optimizer, train_loader, valid_loader, test_loader, epochs=25, log_file='training_log.csv', confusion_matrix_file='confusion_matrix.csv', classification_report_file='classification_report.csv'):
    columns = ['Epoch', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy', 'Test Loss', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test MCC', 'Test CMC', 'Valid Precision', 'Valid Recall', 'Valid F1-Score', 'Valid MCC', 'Valid CMC']
    log_data = []

    for epoch in range(epochs):
        model.train()
        train_loss, valid_loss, test_loss = 0.0, 0.0, 0.0
        train_correct, valid_correct, test_correct = 0, 0, 0
        train_total, valid_total, test_total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                valid_correct += (predicted == labels).sum().item()
                valid_total += labels.size(0)

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        test_loss /= len(test_loader.dataset)
        train_accuracy = train_correct / train_total
        valid_accuracy = valid_correct / valid_total
        test_accuracy = test_correct / test_total

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
        test_precision = precision_score(y_true, y_pred, average='macro')
        test_recall = recall_score(y_true, y_pred, average='macro')
        test_f1 = f1_score(y_true, y_pred, average='macro')
        test_mcc = matthews_corrcoef(y_true, y_pred)
        test_cmc = cohen_kappa_score(y_true, y_pred)

        print("Confusion Matrix:")
        print(cm)
        print("Test Precision:", test_precision)
        print("Test Recall:", test_recall)
        print("Test F1-Score:", test_f1)
        print("Test MCC:", test_mcc)
        print("Test CMC:", test_cmc)

        y_true_valid = []
        y_pred_valid = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true_valid.extend(labels.cpu().numpy())
                y_pred_valid.extend(predicted.cpu().numpy())

        cm_valid = confusion_matrix(y_true_valid, y_pred_valid)
        valid_precision = precision_score(y_true_valid, y_pred_valid, average='macro')
        valid_recall = recall_score(y_true_valid, y_pred_valid, average='macro')
        valid_f1 = f1_score(y_true_valid, y_pred_valid, average='macro')
        valid_mcc = matthews_corrcoef(y_true_valid, y_pred_valid)
        valid_cmc = cohen_kappa_score(y_true_valid, y_pred_valid)

        print("Confusion Matrix (Validation):")
        print(cm_valid)
        print("Valid Precision:", valid_precision)
        print("Valid Recall:", valid_recall)
        print("Valid F1-Score:", valid_f1)
        print("Valid MCC:", valid_mcc)
        print("Valid CMC:", valid_cmc)

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(confusion_matrix_file, index=True)

        # Lưu báo cáo phân loại vào tệp
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(classification_report_file, index=True)

        # Thêm dữ liệu vào bảng nhật ký
        log_data.append([epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy, test_precision, test_recall, test_f1, test_mcc, test_cmc, valid_precision, valid_recall, valid_f1, valid_mcc, valid_cmc])

        # In thông tin tiến trình
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Valid Loss: {valid_loss:.4f} - Valid Accuracy: {valid_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    # Lưu bảng nhật ký vào tệp
    log_df = pd.DataFrame(log_data, columns=columns)
    log_df.to_csv(log_file, index=False)

# Gọi hàm train_model
train_model(model, criterion, optimizer, train_loader, valid_loader, test_loader, epochs=100, log_file='training_log.csv')

torch.save(model.state_dict(), 'resnet152_chicken_gender.pth')