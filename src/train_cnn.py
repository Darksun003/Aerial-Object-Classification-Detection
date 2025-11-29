import tensorflow as tf
import matplotlib.pyplot as plt
from config import IMG_SIZE, EPOCHS_CNN, CLASS_MODELS_DIR, REPORTS_DIR
from data_classification import get_datasets
from models_cnn import build_custom_cnn

def plot_history(history, out_path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    train_ds, val_ds, _, class_names = get_datasets()
    print("Classes:", class_names)

    model = build_custom_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = CLASS_MODELS_DIR / "best_cnn_model.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss"
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_CNN,
        callbacks=callbacks
    )

    plot_history(history, REPORTS_DIR / "accuracy_loss_cnn.png")
    print("Custom CNN training complete. Best model saved to:", checkpoint_path)
