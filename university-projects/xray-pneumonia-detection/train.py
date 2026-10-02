"""
Training script for three models.
Parameters are taken from config.py.
Saves models in the saved_models folder.
Saves training plots (loss/accuracy) in figures/.
"""

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import config
from data_preparation import create_generators
from models import build_basic_cnn,  build_deep_cnn, build_advanced_cnn
from visualize import plot_history

def train_model(model, train_gen, val_gen, model_name, epochs=config.EPOCHS):
    """
    Train the model and save it.
    """
    filepath = os.path.join(config.SAVED_MODELS_DIR, f"{model_name}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early, reduce_lr]
    )

    # Saving training plots
    plot_history(history, model_name)
    return history

if __name__ == "__main__":
    # Creating generators
    train_gen, val_gen, test_gen = create_generators()

    # 1) Basic CNN
    model1 = build_basic_cnn(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    print(model1.summary())
    h1 = train_model(model1, train_gen, val_gen, "model_basic_cnn")

    # 2) Deep CNN
    model2 = build_deep_cnn(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    print(model2.summary())
    h2 = train_model(model2, train_gen, val_gen, "model_deep_cnn")

    # 3) Advanced CNN
    model3 = build_advanced_cnn(input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    h3 = train_model(model3, train_gen, val_gen, "model_advanced_cnn")
    print(model3.summary())

