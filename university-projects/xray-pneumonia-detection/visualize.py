"""
Plots and visualization.
Saves PNG files in the figures folder.
"""
import matplotlib

matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import config

sns.set()


def plot_history(history, model_name):
    """
    Saves loss and accuracy plots.
    """
    out_dir = config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    # accuracy
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, f'{model_name}_accuracy.png'))
    plt.close()

    # loss
    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, f'{model_name}_loss.png'))
    plt.close()


def save_confusion_and_roc(y_true, y_pred_probs, model_name):
    """
    Saves confusion matrix and ROC curve.
    y_pred_probs — probabilities of class 1.
    """
    out_dir = config.FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    # threshold
    y_pred = (y_pred_probs >= config.PREDICTION_THRESHOLD).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(out_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, f'{model_name}_roc.png'))
    plt.close()

    # classification report
    report = classification_report(y_true, y_pred, output_dict=False)
    with open(os.path.join(out_dir, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(report)

    return cm, roc_auc
