"""
Definition of three models:
1) Basic simple CNN
2) Deep CNN (with BatchNorm and Dropout)
3) Advanced CNN (custom ResNet-style blocks with separable convolutions)
All models return a compiled keras.Model
"""

from tensorflow.keras import layers, models, optimizers
import tensorflow as tf
import config

def build_basic_cnn(input_shape=(224,224,3)):
    """
    Simple CNN for a basic level.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="basic_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_deep_cnn(input_shape=(224,224,3)):
    """
    Deeper architecture with BatchNorm and Dropout.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="deep_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_advanced_cnn(input_shape=(224, 224, 3)):
    """
    A custom architecture built from scratch using Residual Connections
    and Separable Convolutions to maximize accuracy.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Entry Block ---
    # Standard Conv2D to capture initial raw features (edges/textures)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # --- Core Block 1 (64 Filters) ---
    previous_block_activation = x  # Save input for skip connection

    x = layers.SeparableConv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.SeparableConv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Downsample using pooling
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Adjust the "skip connection" to match the new shape (1x1 Conv)
    residual = layers.Conv2D(64, (1, 1), strides=2, padding="same")(previous_block_activation)

    # ADD the residual (The "ResNet" magic happens here)
    x = layers.add([x, residual])

    # --- Core Block 2 (128 Filters) ---
    previous_block_activation = x

    x = layers.SeparableConv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.SeparableConv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Adjust residual
    residual = layers.Conv2D(128, (1, 1), strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])

    # --- Core Block 3 (256 Filters) ---
    previous_block_activation = x

    x = layers.SeparableConv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.SeparableConv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Adjust residual
    residual = layers.Conv2D(256, (1, 1), strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])

    # --- Classification Head ---
    x = layers.GlobalAveragePooling2D()(x)

    # Strong regularization to prevent overfitting
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="advanced_resnet_cnn")

    # Using a slightly lower learning rate is often better for custom ResNets
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
