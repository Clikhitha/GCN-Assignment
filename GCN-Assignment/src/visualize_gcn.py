from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

training_file = RESULTS / 'training.csv'
predictions_file = RESULTS / 'predictions.csv'

if not training_file.exists() or not predictions_file.exists():
    raise FileNotFoundError(
        'Could not find results/training.csv and results/predictions.csv. '
        'Run TwoLayerGCN.java first from the project root.'
    )

training = pd.read_csv(training_file)
predictions = pd.read_csv(predictions_file)

# 1. Training loss curve
plt.figure(figsize=(9, 5))
plt.plot(training['epoch'], training['train_loss'], label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('GCN Training Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / 'training_loss.png', dpi=200)
plt.close()

# 2. Validation metrics across epochs
plt.figure(figsize=(9, 5))
plt.plot(training['epoch'], training['val_accuracy'], label='Validation accuracy')
plt.plot(training['epoch'], training['val_macro_f1'], label='Validation macro-F1')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('GCN Validation Performance')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / 'validation_metrics.png', dpi=200)
plt.close()

# 3. Test-set confusion matrix
labels = sorted(set(predictions['actual']).union(set(predictions['predicted'])))
cm = confusion_matrix(predictions['actual'], predictions['predicted'], labels=labels)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(10, 8))
display.plot(ax=ax, xticks_rotation=45, values_format='d', colorbar=False)
ax.set_title('Cora Test Set Confusion Matrix')
fig.tight_layout()
fig.savefig(RESULTS / 'confusion_matrix.png', dpi=200)
plt.close(fig)

# 4. Actual vs predicted class counts
actual_counts = predictions['actual'].value_counts().sort_index()
predicted_counts = predictions['predicted'].value_counts().reindex(actual_counts.index, fill_value=0)
counts = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts})
ax = counts.plot(kind='bar', figsize=(11, 5))
ax.set_title('Actual vs Predicted Class Distribution (Test Set)')
ax.set_xlabel('Paper class')
ax.set_ylabel('Number of papers')
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'class_distribution.png', dpi=200)
plt.close()

print('Visualizations saved in results/:')
for name in ['training_loss.png', 'validation_metrics.png', 'confusion_matrix.png', 'class_distribution.png']:
    print(f' - {name}')
print('\nTest-set classification report:')
print(classification_report(predictions['actual'], predictions['predicted'], zero_division=0))
