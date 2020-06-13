import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_preds, training_type, seed):
    cm_data = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_data)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion Matrix {training_type} - seed: {seed}")

    filename = f"{training_type}_seed{seed}_cm.png"  # ex. cm_lwf_30
    plt.savefig(filename, format='png', dpi=300)
    plt.show()


def plot_accuracy_trend(accuracies, training_type, seed):
    class_batches = np.arange(10, 101, 10)

    plt.figure()
    plt.scatter(class_batches, np.array(accuracies) * 100, zorder=100, c='orange')
    plt.plot(class_batches, np.array(accuracies) * 100)
    plt.xlabel("Number of classes")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 100)
    plt.xticks(list(range(10, 101, 10)))
    plt.yticks(list(range(0, 101, 10)))
    plt.grid()

    filename = f"{training_type}_seed{seed}_acc.png"  # ex. cm_lwf_30
    plt.savefig(filename, format='png', dpi=300)
    plt.show()
