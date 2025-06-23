import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_confusion_matrix(y_true_encoded, y_pred_encoded, label_encoder, maior, title="Matriz de Confusão"):
    y_true_original = label_encoder.inverse_transform(y_true_encoded)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    labels = np.unique(np.concatenate([y_true_original, y_pred_original]))

    cm = confusion_matrix(y_true_original, y_pred_original, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    if (maior):
        fig, ax = plt.subplots(figsize=(12, 8))
        disp.plot(cmap='Blues', values_format='d', ax=ax)
    else:
        disp.plot(cmap='Blues', values_format='d')
    plt.xticks(rotation=90)  # Rotaciona labels do eixo x para vertical
    plt.title(title)
    plt.tight_layout()       # Ajusta layout para não cortar labels
    plt.show()

def classification_metrics(y_true_encoded, y_pred_encoded, label_encoder):
    y_true_original = label_encoder.inverse_transform(y_true_encoded)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
    print(classification_report(y_true_original, y_pred_original))
