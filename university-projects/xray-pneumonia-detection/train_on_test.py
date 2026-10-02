"""
EXPERIMENT ONLY: continues training a saved model on the test split.

A model produced by this script has seen the test images, so its test-set
scores are not a fair measure of generalization. The metrics reported in the
README come from models trained only on the training split (train.py + evaluate.py).
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
tf.config.run_functions_eagerly(True)

def create_test_generator(img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, seed=config.SEED):
    """
    Generator for the test dataset (used for additional training).
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_gen = datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=seed,
        shuffle=True
    )

    return test_gen


def train_on_test():
    """
    Load the saved model and continue training on the test data.
    """
    model_path = os.path.join(config.SAVED_MODELS_DIR, "model_deep_cnn.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found!")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create test data generator
    test_gen = create_test_generator()

    # Callbacks
    save_path = os.path.join(config.SAVED_MODELS_DIR, "model_deep_cnn_fully_trained.h5")
    checkpoint = ModelCheckpoint(save_path, monitor='accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(
        monitor='accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    # Additional training
    print("\nStarting additional training on the test data...")
    history = model.fit(
        test_gen,
        epochs=20,
        callbacks=[checkpoint, early, reduce_lr]
    )

    print(f"\nFully trained model saved to: {save_path}")


if __name__ == "__main__":
    train_on_test()
