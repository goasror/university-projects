"""
Project configuration parameters.
Setup for paths, hyperparameters, and bot token.
"""

import os

# Path to the extracted Kaggle dataset
# Expected structure: train/NORMAL, train/PNEUMONIA, val/..., test/...
DATA_DIR = os.path.join(os.getcwd(), "chest_xray")

# Folders inside DATA_DIR
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Image parameters
IMG_SIZE = (224, 224)  # can be changed to 128 or 224
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 2  # NORMAL / PNEUMONIA

# Training
EPOCHS = 20
LEARNING_RATE = 1e-4

# Folder for saved models and visualizations
SAVED_MODELS_DIR = os.path.join(os.getcwd(), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

FIGURES_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Telegram bot token: read from the environment, never commit it.
# export TELEGRAM_BOT_TOKEN="123456:ABC..."  (see .env.example)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Model the bot serves (file name inside saved_models/)
BOT_MODEL_FILE = os.environ.get("BOT_MODEL_FILE", "model_advanced_cnn.h5")

# Predictions: threshold for binary decision (can be changed)
PREDICTION_THRESHOLD = 0.5
