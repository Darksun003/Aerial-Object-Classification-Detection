from keras import layers, Model
from keras.applications import MobileNetV2
from keras.models import Sequential


def build_custom_cnn(input_shape=(224, 224, 3)):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=input_shape),

        layers.Conv2D(32, 3, activation='relu', padding="same"),
        layers.MaxPool2D(),

        layers.Conv2D(64, 3, activation='relu', padding="same"),
        layers.MaxPool2D(),

        layers.Conv2D(128, 3, activation='relu', padding="same"),
        layers.MaxPool2D(),

        layers.Dropout(0.3),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_transfer_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # freeze feature extractor

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1. / 255)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)
