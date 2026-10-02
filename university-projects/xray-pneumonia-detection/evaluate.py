"""
Model evaluation on the test dataset.
Loads a saved model, performs predictions, and saves visualizations.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from data_preparation import create_generators
import config
from visualize import save_confusion_and_roc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import keras
keras.config.enable_unsafe_deserialization()

def evaluate_model(model_path, model_name):
    """
    Load the model and evaluate it on the test_generator.
    """
    print(f"Evaluating model {model_name}...")
    model = load_model(model_path, safe_mode=False)
    _, _, test_gen = create_generators()

    # Predictions
    # test_gen has shuffle=False, so label order is preserved
    y_true = test_gen.classes  # 0/1
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    preds_proba = model.predict(test_gen, steps=steps, verbose=1).ravel()
    # trim extra predictions (may occur if last batch is incomplete)
    preds_proba = preds_proba[:len(y_true)]

    # save metrics
    cm, roc_auc = save_confusion_and_roc(y_true, preds_proba, model_name)
    preds = (preds_proba >= config.PREDICTION_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

    # Save metrics to a file
    out_dir = config.FIGURES_DIR
    with open(os.path.join(out_dir, f'{model_name}_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nAUC: {roc_auc:.4f}\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc
    }

if __name__ == "__main__":
    # Iterate through saved models, if any exist
    models_dir = config.SAVED_MODELS_DIR
    for fname in os.listdir(models_dir):
        if fname.endswith(".h5"):
            path = os.path.join(models_dir, fname)
            name = fname.replace(".h5", "")
            evaluate_model(path, name)
