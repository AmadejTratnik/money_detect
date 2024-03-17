import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

def save_confusion_matrices(confusion_matrices, output_file='images/confusion_matrices.png'):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes.flatten()[i]
        ax.set_title(model_name)
        sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'], ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_file)



def save_roc_curves(roc_curves, output_file='images/roc_curves.png'):
    plt.figure(figsize=(8, 6))
    for model_name, (fpr, tpr, _) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)