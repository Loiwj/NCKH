import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import datetime
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)


def preprocess_image(image_path, target_size=(300, 300)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    return image_array


# Thư mục chứa dữ liệu
data_dir = "./Detect_chicken_sex_V3"

# List các tên lớp (tên thư mục trong data_dir)
class_names = os.listdir(data_dir)
num_classes = len(class_names)

# Load dữ liệu từ thư mục
inputs = []
targets = []

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
NUM_CLASSES = 2
EPOCHS = 40
for class_index, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        inputs.append(preprocess_image(image_path))
        targets.append(class_index)

inputs = np.array(inputs)
targets = np.array(targets)


# Định nghĩa các tham số của K-fold Cross Validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
acc_per_fold = []
loss_per_fold = []


def build_model():
    base_model = EfficientNetB3(  # Sử dụng EfficientNetB3
        weights="imagenet",
        include_top=False,
        input_shape=(300, 300, 3),
    )

    for layer in base_model.layers:
        layer.trainable = True

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(2048, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(name="val_precision"),
            tf.keras.metrics.Recall(name="val_recall"),
        ],
    )

    return model


# Chuyển đổi nhãn thành one-hot encoding
targets_one_hot = to_categorical(targets, num_classes)

checkpoint = ModelCheckpoint(
    "best_model_EfficientNetB3_v1_tangcuong.keras",
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)


class MetricsLogger(Callback):
    def __init__(self, log_file, X_val, y_val, fold_no, log_file_prefix):
        super().__init__()
        self.log_file = log_file
        self.fold_no = fold_no
        self.log_file_prefix = log_file_prefix
        self.epoch_count = 0
        self.X_val = X_val
        self.y_val = y_val
        self.header_written = False

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, "a") as f:
            if not self.header_written:
                f.write(
                    "Epoch\tTrain loss\tTrain accuracy\tval_loss\tval_accuracy\tval_recall\tval_precision\tvalid_F1-Score\n"
                )
                self.header_written = True
            y_true = np.argmax(self.y_val, axis=1)
            y_pred = np.argmax(self.model.predict(self.X_val), axis=1)
            f1 = f1_score(y_true, y_pred, average="weighted")
            f.write(
                f"{epoch+1}\t{logs['loss']:.5f}\t{logs['accuracy']:.5f}\t{logs['val_loss']:.5f}\t{logs['val_accuracy']:.5f}\t{logs['val_recall']:.5f}\t{logs['val_precision']:.5f}\t{f1:.5f}\n"
            )

        confusion_matrix_file = f"{self.log_file_prefix}_fold{self.fold_no}.txt"
        save_confusion_matrix_append(y_true, y_pred, class_names, confusion_matrix_file)

    def on_train_end(self, logs=None):
        print(f"Confusion matrix for fold {self.fold_no} has been saved.")


def save_confusion_matrix_append(y_true, y_pred, class_names, file_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    with open(file_path, "a") as f:
        df_cm.to_csv(f, sep="\t", mode="a")


def save_classification_report(y_true, y_pred, class_names, file_path):
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(file_path, "a") as f:
        f.write(report)


for fold_no, (train_indices, test_indices) in enumerate(
    kfold.split(inputs, targets), 1
):
    X_train, X_val = inputs[train_indices], inputs[test_indices]
    y_train, y_val = targets_one_hot[train_indices], targets_one_hot[test_indices]

    # Reset model mỗi lần chạy fold mới
    
    model = build_model()
    model.build((None, *IMG_SIZE, 3))
    model.summary()
    # Tính toán confusion matrix cho tập train trước khi tăng cường
    y_train_pred_before_augmentation = np.argmax(model.predict(X_train), axis=1)
    y_train_true = np.argmax(y_train, axis=1)
    confusion_matrix_train_before_augmentation = confusion_matrix(
        y_train_true, y_train_pred_before_augmentation
    )
    print("Confusion matrix for train data before augmentation:")
    print(confusion_matrix_train_before_augmentation)
    # Khởi tạo MetricsLogger mới cho mỗi fold
    metrics_logger = MetricsLogger(
        f"metrics_EfficientNetB3_v1_tangcuong_fold_{fold_no}.log",
        X_val,
        y_val,
        fold_no,
        f"confusion_matrix_EfficientNetB3_v1_tangcuong",
    )
    # Khởi tạo ImageDataGenerator để áp dụng tăng cường dữ liệu cho tập huấn luyện của fold hiện tại
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=False,
        fill_mode="nearest",
    )
    # Tạo ra dữ liệu augmented từ dữ liệu train
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    # Tính toán confusion matrix cho tập train sau khi tăng cường
    y_train_pred_after_augmentation = np.argmax(model.predict(train_generator), axis=1)
    confusion_matrix_train_after_augmentation = confusion_matrix(
        y_train_true, y_train_pred_after_augmentation
    )
    print("Confusion matrix for train data after augmentation:")
    print(confusion_matrix_train_after_augmentation)

    # Lưu confusion matrix vào file
    np.savetxt(
        "confusion_matrix_train_before_augmentation.txt",
        confusion_matrix_train_before_augmentation,
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        "confusion_matrix_train_after_augmentation.txt",
        confusion_matrix_train_after_augmentation,
        fmt="%d",
        delimiter="\t",
    )
    # Huấn luyện mô hình với dữ liệu tăng cường của fold hiện tại
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint, metrics_logger],
        validation_data=(X_val, y_val),
    )

    # Đánh giá mô hình trên dữ liệu kiểm tra của fold hiện tại
    scores = model.evaluate(
        inputs[test_indices], targets_one_hot[test_indices], verbose=1
    )
    print(
        f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
    )

    # Tính toán các metric
    y_pred = model.predict(inputs[test_indices])
    y_pred = np.argmax(y_pred, axis=1)

    save_classification_report(
        targets[test_indices],
        y_pred,
        class_names,
        f"classification_report_EfficientNetB3_v1_tangcuong.txt",
    )
    # Clear the session to free up memory after each fold
    tf.keras.backend.clear_session()
