"""
Data loading and preprocessing script.
Uses Keras ImageDataGenerator for simplicity.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def create_generators(img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, seed=config.SEED):
    """
    Create generators for train/val/test datasets.
    Basic augmentation added for the training set.
    """
    # Normalize to [0,1]
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        seed=seed
    )

    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    tg, vg, tg2 = create_generators()
    print("Train classes:", tg.class_indices)
