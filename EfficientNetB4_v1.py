class MetricsLogger(Callback):
    def __init__(self, log_file, fold_no):
        super().__init__()
        self.log_file = log_file
        self.fold_no = fold_no
        self.header_written = False

    def on_epoch_end(self, epoch, logs=None, y_true=None, y_pred=None, class_names=None):
        with open(self.log_file, "a") as f:
            if not self.header_written:
                f.write(
                    "Epoch\tTrain loss\tTrain accuracy\tval_loss\tval_accuracy\tval_recall\tval_precision\tvalid_MCC\tvalid_CMC\tvalid_F1-Score\n"
                )
                self.header_written = True
            f.write(
                f"{epoch+1}\t{logs['loss']:.5f}\t{logs['accuracy']:.5f}\t{logs['val_loss']:.5f}\t{logs['val_accuracy']:.5f}\t{logs['val_recall']:.5f}\t{logs['val_precision']:.5f}\t{logs['val_recall']:.5f}\t{logs['val_precision']:.5f}\n"
            )

        confusion_matrix_file = f"confusion_matrix_fold{self.fold_no}.txt"
        save_confusion_matrix_append(y_true, y_pred, class_names, confusion_matrix_file)

    def on_train_end(self, logs=None):
        print(f"Confusion matrix for fold {self.fold_no} has been saved.")

# Now you can create instances of MetricsLogger
metrics_loggers = [MetricsLogger(log_file, fold_no) for fold_no, log_file in enumerate(metrics_log_files, 1)]
