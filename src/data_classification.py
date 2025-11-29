import tensorflow as tf
from config import CLASS_DATASET_DIR, IMG_SIZE, BATCH_SIZE

def get_datasets():
    train_dir = CLASS_DATASET_DIR / "train"
    val_dir = CLASS_DATASET_DIR / "valid"
    test_dir = CLASS_DATASET_DIR / "test"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    class_names = train_ds.class_names

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
