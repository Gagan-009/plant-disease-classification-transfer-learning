import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(y_true, y_pred):
    """
    Computes overall accuracy and macro-averaged precision, recall, and F1-score.
    Uses zero_division=0 to handle classes with zero predicted/true instances gracefully in small batches.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1)
    }
