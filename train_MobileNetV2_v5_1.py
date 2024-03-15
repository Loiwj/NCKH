import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import csv
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold


# Define dataset class
class ChickenDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = os.path.basename(os.path.dirname(img_path))  # Get label from parent directory
        label = 1 if label == 'cock' else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Define model training function
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=25, log_file='training_log.csv'):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss',
                        'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score',
                        'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1-Score',
                        'Confusion Matrix'])

    for epoch in range(epochs):
        model.train()
        train_loss, test_loss = 0.0, 0.0
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
                            test_accuracy, test_precision, test_recall, test_f1_score,
                            train_accuracy, train_precision, train_recall, train_f1_score,
                            cm_test])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image paths
image_paths = []
for root, _, files in os.walk('./Detect_chicken_sex_V3'):
    for file in files:
        if file.endswith('.jpg'):
            image_paths.append(os.path.join(root, file))

# Define k-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

# List to store test accuracies for each fold
test_accuracies = []

# Perform k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(image_paths)):
    print(f'Fold {fold+1}/{k_folds}')
    
    # Create data loaders for training and validation folds
    train_fold = [image_paths[i] for i in train_index]
    train_dataset_fold = ChickenDataset(train_fold, transform=transform)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=16, shuffle=True)
    
    test_fold = [image_paths[i] for i in test_index]
    test_dataset_fold = ChickenDataset(test_fold, transform=transform)
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=16, shuffle=False)
    
    # Initialize MobileNetV2 model
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, criterion, optimizer, train_loader_fold, test_loader_fold, epochs=50)
    
    # Evaluate the model on the test fold
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader_fold:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_accuracy = correct_test / total_test
    test_accuracies.append(test_accuracy)
    print(f'Test Accuracy for Fold {fold+1}: {test_accuracy:.5f}')

# Print overall results
print(f'Average Test Accuracy: {sum(test_accuracies) / len(test_accuracies):.5f}')
torch.save(model.state_dict(), 'chicken_gender.pth')